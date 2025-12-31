import torch
from myevo2 import Evo2

# 测试使用本地模型路径（默认）
print("=" * 50)
print("Testing Evo2 with local model (default)")
print("=" * 50)

# 使用默认本地路径加载模型
evo2_model = Evo2('evo2_1b_base')

sequence = 'ACGT'
print(f"\nInput sequence: {sequence}")

# Tokenize 序列
tokenized = evo2_model.tokenizer.tokenize(sequence)
print(f"Tokenized: {tokenized}")

input_ids = torch.tensor(
    tokenized,
    dtype=torch.int,
).unsqueeze(0).to('cuda:0')

print(f"Input IDs shape: {input_ids.shape}")

# 前向传播
outputs, _ = evo2_model(input_ids)
logits = outputs[0]

print(f'\nLogits shape (batch, length, vocab): {logits.shape}')
print(f'Logits sample (first 5 values): {logits[0, 0, :5]}')

# 测试序列评分
print("\n" + "=" * 50)
print("Testing sequence scoring")
print("=" * 50)

test_sequences = ['ACGT', 'ATCG', 'GCTA']
scores = evo2_model.score_sequences(test_sequences)
print(f"Sequences: {test_sequences}")
print(f"Scores: {scores}")

output = evo2_model.generate(prompt_seqs=["AATGGTCTATACCCGTGGTCTAGGAAATCGCTTGCAATGTCCTTTATTACCGCATCGTCTTTGCTCTCCGCCGTTACATAGGCTAAACCGCGTTCCCATAGAGCCCCCGTAAACCACGATTTACCGCAATTACCGACTTTGTCATACCACAACACTACCTCACGGTCGTTTGTCCTTCTAAGGGCTTTTATGACCCTTTCCTGAGCCTCTCTGTATTTCCCGAATCTTTGTATGATGTTTTCCACTCTATCATCGCTCTTTACATAGTGTCCTTCTTTGGTTTCGTAGTCACTCCATTTATCGGATTTCTCGATATGTGATTTCTTTTCTATGGAGAATTCTCCCCATCCTGTTGCTCCGAACCCTATGAATTCCTTTCTTGCCGTGAATTCGCTGAAATGCTCGAAGAATTGATCGTCTCCGCATTCTACTCTCCATTGACAATGCTTGTATCCTCCTTTTCCTGTTTCTATTCCTAATGTCCATCTTTTGCAGTCGTAGAGATTCTTAAA"], n_tokens=512, temperature=1.0, top_k=4)
print(f"Output: {output}")
# print("\n" + "=" * 50)
# print("Test completed successfully!")
# print("=" * 50)