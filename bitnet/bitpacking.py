import torch
from torch import uint8, int8, Tensor

class BitPack:
    # 1.58-bit -> adapted from HQQ 
    ################################################
    #
    @staticmethod
    def check_158bit(W_q: Tensor) -> Tensor:
        """
        check if 2-bit quantization is possible
        """
        assert W_q.dtype == int8, f"Expected uint8 but got {W_q.dtype}"
        assert W_q.dim() == 2, f"Need a 2D array f"
        assert len(W_q) % 4 == 0, f"Expected len(W_q) to be multiple of 4 but got {len(W_q)}"
        assert W_q.max() <= 1, f"Expected W_q.max() <= 1 but got {W_q.max()}"
        assert W_q.min() >= -1, f"Expected W_q.min() >= 0 but got {W_q.min()}"
        
        W_q_Dq = BitPack.unpack_158bit_u8(BitPack.pack_158bit_u8(W_q))
        assert torch.allclose(W_q, W_q_Dq), f"Expected W == W_dequant but got {W_q} != {W_q_Dq}"
        return True
    
    @staticmethod
    def pack_158bit_u8(W_q: Tensor) -> Tensor:  # uint8 > uint8/4
        """
        pack a stack of 4x ternary {-1,0,1} weights into
        2-bit weights, hidden as a single uint8 tensor.
        Theoretically, could also pack 5x ternary weights
        """
        W_q = (W_q + 1).to(uint8)
        _step = int(len(W_q) / 4)

        return (
            W_q[:_step] << 6
            | W_q[_step : 2 * _step] << 4
            | W_q[2 * _step : 3 * _step] << 2
            | W_q[3 * _step :]
        ) 

    @staticmethod
    def unpack_158bit_u8(W_q: Tensor) -> Tensor:
        _step = W_q.shape[0]
        tmp = torch.empty([4 * _step, W_q.shape[1]], dtype=uint8, device=W_q.device)

        tmp[0 * _step : 1 * _step] = (W_q & 0b11000000) >> 6
        tmp[1 * _step : 2 * _step] = (W_q & 0b00110000) >> 4
        tmp[2 * _step : 3 * _step] = (W_q & 0b00001100) >> 2
        tmp[3 * _step : 4 * _step] = W_q & 0b00000011

        return tmp.to(int8) - 1

if __name__ == "__main__":
    # test 1.58-bit
    W = torch.randint(-1, 2, (8, 8)).to(torch.int8)
    W_q = BitPack.pack_158bit_u8(W)
    W_Dq = BitPack.unpack_158bit_u8(W_q)
    assert torch.allclose(W, W_Dq), f"Expected W == W_Dq but got {W} != {W_Dq}"
    print("1.58-bit test passed")