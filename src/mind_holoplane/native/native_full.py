
from torch_scatter import segment_sum_csr
import torch
import numpy as np
import torch.sparse as ts
from linear_operator.utils.linear_cg import linear_cg

@torch.no_grad()
def isotropic_elastic_tensor_parallel(E,nu):
    Lambda = nu / (1. + nu) / (1 - 2. * nu)
    Mu = 1. / (2.*(1. + nu))
    dC_dE=torch.tensor([
		[Lambda + 2 * Mu, Lambda, Lambda, 0, 0, 0],
		[Lambda, Lambda + 2 * Mu, Lambda, 0, 0, 0],
		[Lambda, Lambda, Lambda + 2 * Mu, 0, 0, 0],
		[0, 0, 0, Mu, 0, 0],
		[0, 0, 0, 0, Mu, 0],
		[0, 0, 0, 0, 0, Mu]]).to(E.device)

    return torch.einsum('i,jk->ijk',E,dC_dE)

@torch.no_grad()
def isotropic_elastic_tensor(E, v):
    Lambda = v / (1. + v) / (1 - 2. * v)*E
    Mu = 1. / (2.*(1. + v))*E

    return torch.as_tensor([
		[Lambda + 2 * Mu, Lambda, Lambda, 0, 0, 0],
		[Lambda, Lambda + 2 * Mu, Lambda, 0, 0, 0],
		[Lambda, Lambda, Lambda + 2 * Mu, 0, 0, 0],
		[0, 0, 0, Mu, 0, 0],
		[0, 0, 0, 0, Mu, 0],
		[0, 0, 0, 0, 0, Mu]])


class homogenization:


    def __init__(self,cache_file, nelx, nely, nelz, lx, ly, lz,device) -> None:

        cache = np.load(cache_file)
        self.partial_N = torch.from_numpy(cache['partial_N']).float().to(device)
        self.weights = torch.from_numpy(cache['weight']).float().to(device)

        self.__K_indices = None
        self.__F_indices = None

        self.__anchor_indices = None
        self.__K_mask = None
        self.__F_mask = None

        self.__K_sortidx = None
        self.__K_ptr = None

        self.__F_sortidx = None
        self.__F_ptr = None

        self.__lx = lx
        self.__ly = ly
        self.__lz = lz
        self.__nelx = nelx
        self.__nely = nely
        self.__nelz = nelz
        self.device=device

        nel = nelx * nely * nelz
        self.nel=nel
        nodeidx = torch.arange(0, self.nel,device=self.device).view(nelx, nely, nelz)

        index = torch.as_tensor([0],device=self.device)
        nodeidx = torch.cat(
            (nodeidx, torch.index_select(nodeidx, 0, index)), 0)
        nodeidx = torch.cat(
            (nodeidx, torch.index_select(nodeidx, 1, index)), 1)
        nodeidx = torch.cat(
            (nodeidx, torch.index_select(nodeidx, 2, index)), 2)

        node_list = [nodeidx[0:nelx, 0:nely, 0:nelz].reshape((nel, 1)),
                     nodeidx[1:nelx + 1, 0:nely, 0:nelz].reshape((nel, 1)),
                     nodeidx[1:nelx + 1, 1:nely + 1, 0:nelz].reshape((nel, 1)),
                     nodeidx[0:nelx, 1:nely + 1, 0:nelz].reshape((nel, 1)),
                     nodeidx[0:nelx, 0:nely, 1:nelz + 1].reshape((nel, 1)),
                     nodeidx[1:nelx + 1, 0:nely, 1:nelz + 1].reshape((nel, 1)),
                     nodeidx[1:nelx + 1, 1:nely + 1,
                     1:nelz + 1].reshape((nel, 1)),
                     nodeidx[0:nelx, 1:nely + 1, 1:nelz + 1].reshape((nel, 1))]

        self.__cellidx = torch.zeros(
            8, 3, nel, device=self.device, dtype=torch.int64)
        self.__cellseq = torch.zeros(nel, 8, device=self.device, dtype=torch.int64)

        for i in range(8):
            self.__cellidx[i] = self.index2xyz(node_list[i])
            self.__cellseq[:, i] = node_list[i].view(-1)

        self.__celldof=self.__cellidx.permute(2,0,1).reshape(nel,-1)

        self.__nodeidx = nodeidx

        self.volume = self.__lx * self.__ly * self.__lz

        self.anchor()
        self.indices()
        self.shape_matrix()

    @torch.no_grad()
    def stiffness_force(self,elastic_tensor):
        """compute the stiffness and force of a hex element

        Args:
            elastic_tensor ([type]): 6 * 6

        Returns:
            stiffness [type]: 24*24
            force [type] 24*6
        """
        # Gaussian integration
        # k=B^T @ C @ B *w
        stiffness = (self.B.transpose(1, 2)@elastic_tensor@self.B*self.w).sum(0)
        force = (self.B.transpose(1, 2)@elastic_tensor*self.w).sum(0)
        return stiffness, force

    @torch.no_grad()
    def assembly(self,E,nu):
        # C=isotropic_elastic_tensor_parallel(E,nu).float()
        # vK,vF=vmap(self.stiffness_force)(C)

        vK=torch.einsum('i,jk->ijk',E,self.K0)
        vF=torch.einsum('i,jk->ijk',E,self.F0)

        # new anchor by set F(anchor)->0
        anchor_indices=torch.as_tensor(self.__anchor_indices,dtype=torch.int64)

        for idx in range(self.__K_mask.shape[0]):
            vK[anchor_indices[idx], :, :] =  vK[anchor_indices[idx], :, :] *  self.__K_mask[idx, :, :]
            vF[anchor_indices[idx], :, :] =  vF[anchor_indices[idx], :, :] *  self.__F_mask[idx, :, :]

        vK=self.symcoalesce( vK.contiguous().view(-1))
        F=self.coalesce( vF.contiguous().view(-1)).view(-1,6)
        K = torch.sparse_coo_tensor(self.__K_indices, vK, (3 * self.nel, 3 * self.nel), device=self.device).coalesce()
        # F=torch.sparse_coo_tensor(self.__F_indices, vF, (3 * self.nel, 6), device=self.device).coalesce()

        return K,F


    def solve(self,K,F,tol=1e-5, maxit=10000):
        def Kmm(rhs): return ts.mm(K, rhs)
        X=linear_cg(Kmm, F, tolerance=tol, max_iter=maxit)
        self.error=(Kmm(X)-F).norm()/F.norm()
        return X

    @torch.no_grad()
    def homogenized(self, voxel, U):
        solid_seq = self.__cellseq[voxel.type(
            torch.bool).contiguous().view(-1), :]
        n = solid_seq.shape[0]

        index_u = torch.empty(n, 8, 3, dtype=torch.int64, device=self.device)
        for i in range(3):
            index_u[:, :, i] = 3 * solid_seq + i
        index_u = index_u.contiguous().view(-1)

        u = U[index_u, :].contiguous().view(n, 24, 6)
        del index_u

        CH = torch.zeros(6, 6, dtype=self.K0.dtype, device=self.device)
        L = self.U0-u
        L=L.float()
        CH = torch.einsum('bij,ik,bkl-> jl',L,self.K0,L)
        return 1 / self.volume * CH

    @torch.no_grad()
    def homogenized_grad(self,U):
        index_u = torch.empty(self.nel, 24, dtype=torch.int64, device=self.device)
        for i in range(3):
            index_u[:, 3*torch.arange(8)+i] = 3 * self.__cellseq + i
        index_u = index_u.contiguous()

        u = U[index_u, :].contiguous()

        L=self.U0-U[index_u]
        del index_u
        return 1/self.volume*torch.einsum('bij,ik,bkl->bjl',L,self.K0,L)

    @torch.no_grad()
    def macro_deformation(self,elastic_tensor):
        idx = torch.ones(24, dtype=torch.bool, device=self.device)
        idx[[0, 1, 2, 4, 5, 11]] = False

        self.K0, self.F0 = self.stiffness_force(elastic_tensor)

        self.U0 = torch.zeros(24, 6, dtype=elastic_tensor.dtype, device=self.device)
        self.U0[idx, :] = torch.inverse(self.K0[idx, :][:, idx])@self.F0[idx, :]


    @torch.no_grad()
    def shape_matrix(self):
        dx = self.__lx / self.__nelx / 2
        dy = self.__ly / self.__nely / 2
        dz = self.__lz / self.__nelz / 2

        hex = torch.tensor(
            [[-dx, dx, dx, -dx, -dx, dx, dx, -dx], [-dy, -dy, dy, dy, -dy, -dy, dy, dy],
             [-dz, -dz, -dz, -dz, dz, dz, dz, dz]], device=self.device).t()

         # coumpute jacobian 27 * 3 * 3
        J = torch.matmul(self.partial_N, hex)

        # compute weight 27*1
        self.w = (J.det().unsqueeze(1)*self.weights).unsqueeze(2)

        # compute the inverse of partial N 27 * 3 * 8
        inv_J = J.inverse()
        inv_N = (inv_J@self.partial_N)

        # assigned local geometry matrix B 27*6*24
        B = inv_N.unsqueeze(1).repeat(1, 6, 1, 1)*0
        idx_B = torch.as_tensor([[0, 1, 2, 3, 3, 4, 4, 5, 5], [
                                0, 1, 2, 0, 1, 1, 2, 0, 2]], dtype=torch.int64, device=hex.device)
        idx_inv_N = torch.as_tensor(
            [0, 1, 2, 1, 0, 2, 1, 2, 0], dtype=torch.int64, device=hex.device)
        B[:, idx_B[0], idx_B[1], :] = inv_N[:, idx_inv_N, :]

        self.B = B.transpose(2, 3).contiguous().reshape(27, 6, 24)

    @torch.no_grad()
    def index2xyz(self, index):
        x = index.div(self.__nely * self.__nelz, rounding_mode='floor')
        temp = index .remainder(self.__nely * self.__nelz)
        y = temp.div(self.__nely,rounding_mode='floor')
        z = temp.remainder( self.__nely)
        xyz = torch.cat((x, y, z), 1)
        return xyz.t()

    @torch.no_grad()
    def anchor(self, index=0):
        anchor = self.__nodeidx[self.__cellidx[index, 0, 0],
                                self.__cellidx[index, 1, 0], self.__cellidx[index, 2, 0]]

        mask = torch.eq(self.__cellseq, anchor)

        anchor_index = torch.arange(
            0, self.__nelx*self.__nely*self.__nelz, device=self.device)
        anchor_cell = anchor_index.masked_select(mask.sum(1).type(torch.bool))
        anchor_mask = mask[anchor_cell, :]

        if anchor_mask.dim() == 1:
            anchor_mask.unsqueeze_(0)

        anchor_mask = ~anchor_mask.unsqueeze(2).repeat(
            1, 1, 3).reshape(-1, 24).unsqueeze(2)
        anchor_mask = torch.as_tensor(
            anchor_mask, dtype=torch.float64, device=self.device)
        K_mask = anchor_mask.bmm(anchor_mask.transpose(1, 2))
        K_diag = torch.eye(24, 24, dtype=K_mask.dtype, device=self.device).unsqueeze(
            0).repeat(K_mask.shape[0], 1, 1)
        K_mask = torch.logical_or(K_mask, K_diag)
        F_mask = anchor_mask.repeat(1, 1, 6)

        self.__anchor_indices = torch.as_tensor(anchor_cell,dtype=torch.float,device=self.device)
        self.__K_mask = torch.as_tensor(K_mask,dtype=torch.float,device=self.device)
        self.__F_mask = torch.as_tensor(F_mask,dtype=torch.float,device=self.device)

        # return anchor_cell, K_mask, F_mask

    @torch.no_grad()
    def indices(self):
        n = self.__nelx * self.__nely * self.__nelz
        dof_indices = torch.empty(
            n, 8, 3, dtype=self.__cellseq.dtype, device=self.device)
        for i in range(3):
            dof_indices[:, :, i] = 3*self.__cellseq+i
        dof_indices = torch.as_tensor(
            dof_indices, dtype=torch.float, device= self.device).view(-1, 24).unsqueeze(2)

        # torch-sparse_solver version
        Kij = torch.zeros(2, 24 * 24 * n, device= self.device)
        temp = torch.ones(
            n, 24, 1, dtype=torch.float, device= self.device)
        Kij[0, :] = dof_indices.bmm(temp.transpose(1, 2)).contiguous().view(-1)
        Kij[1, :] = temp.bmm(dof_indices.transpose(1, 2)).contiguous().view(-1)

        Fij = torch.zeros(2, 24 * 6 * n, device= self.device)
        temp_F = torch.ones(n, 6, 1, dtype=torch.float, device= self.device)
        Fij[0, :] = dof_indices.bmm(
            temp_F.transpose(1, 2)).contiguous().view(-1)
        Fij[1, :] = torch.arange(0, 6, device= self.device).unsqueeze(0).unsqueeze(
            0).repeat(n, 24, 1).contiguous().view(-1)

        self.__K_indices = torch.as_tensor(Kij, dtype=torch.int64)
        self.__F_indices = torch.as_tensor(Fij, dtype=torch.int64)



        # def set_coalesce(self):
        nd = 3*self.__nelx * self.__nely * self.__nelz
        # coalesce+symmetry
        sorted_K,self.__K_sortidx=torch.cat([self.__K_indices[0]*nd+self.__K_indices[1], self.__K_indices[1]*nd+self.__K_indices[0]]).sort()
        sorted_K=torch.cat([-torch.ones(1,dtype=sorted_K.dtype,device=self.device),sorted_K])
        mask = sorted_K[1:] > sorted_K[:-1]
        self.__K_indices = self.__K_indices.repeat(1, 2)[:, self.__K_sortidx][:, mask]

        # print(self.__K_indices.shape)
        self.__K_ptr = mask.nonzero().flatten()
        self.__K_ptr = torch.cat([self.__K_ptr, self.__K_ptr.new_full((1, ),  mask.numel())])


        sorted_F,self.__F_sortidx=(self.__F_indices[0]*6+self.__F_indices[1]).sort()
        sorted_F=torch.cat([-torch.ones(1,dtype=sorted_F.dtype,device=self.device),sorted_F])
        mask = sorted_F[1:] >sorted_F[:-1]
        self.__F_indices = self.__F_indices[:, self.__F_sortidx][:, mask]
        self.__F_ptr = mask.nonzero().flatten()
        self.__F_ptr = torch.cat(
            [self.__F_ptr, self.__F_ptr.new_full((1, ),  mask.numel())])


    @torch.no_grad()
    def symcoalesce(self, value):
        return segment_sum_csr(value.repeat(2)[self.__K_sortidx]*0.5, self.__K_ptr)

    @torch.no_grad()
    def coalesce(self, value):
        value_ = segment_sum_csr(value[self.__F_sortidx], self.__F_ptr)
        return value_
