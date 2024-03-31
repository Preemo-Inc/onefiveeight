import torch
from bitpacking import BitPack

def test_bitpacking_2():
    W = torch.randint(-1, 2, (1280,1280), dtype=torch.int8)
    BitPack.check_2bit(W)
    print("PASSED")
    
def test_compile_bitpacking_2():
    """test not graph breakage"""
    @torch.compile
    def compile_fn(W):
        return BitPack.unpack_158bit_u8(BitPack.pack_158bit_u8(W))
    
    W = torch.randint(-1, 2, (1280,492), dtype=torch.int8)
    compile_fn(W)
    explaination =  torch._dynamo.explain(compile_fn, W) 
    assert explaination.graph_break_count == 0, f"Expected 0 but got {explaination.graph_break_count}"
    print("COMPILED PASSED")

if __name__ == "__main__":
    test_bitpacking_2()
    test_compile_bitpacking_2()
    