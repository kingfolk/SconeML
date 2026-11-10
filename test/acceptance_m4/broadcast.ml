(* @case shape-0 *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: shaped-f32-cpu
    profile: f32-ftz
    arguments:
      - name: a
        dtype: f32
        shape: [2, 3]
        values: [0x7d8a8b93, 0x716f9bd6, 0xaf17213d, 0x96cef178, 0xeddf0677, 0xcef4b16a]
      - name: b
        dtype: f32
        shape: [3]
        values: [0x7da3f1a0, 0x88a7f87f, 0x5e602bd2]
      - name: c
        dtype: f32
        shape: [2, 1]
        values: [0x7dbd57ad, 0x9fe05528]
  - name: metal
    pipeline: shaped-f32-metal
    profile: f32-ftz
    threads-per-group: 4
    arguments:
      - name: a
        dtype: f32
        shape: [2, 3]
        values: [0x7d8a8b93, 0x716f9bd6, 0xaf17213d, 0x96cef178, 0xeddf0677, 0xcef4b16a]
      - name: b
        dtype: f32
        shape: [3]
        values: [0x7da3f1a0, 0x88a7f87f, 0x5e602bd2]
      - name: c
        dtype: f32
        shape: [2, 1]
        values: [0x7dbd57ad, 0x9fe05528]
*)
(* @assert:cpu:result
[0x7f800000, 0x7dbd57ad, 0x7dbd57ad, 0xd5048712, 0x6ddf0677, 0xedd6451f]
*)
(* @assert:cpu:shape
[2, 3]
*)
(* @assert:metal:result
[0x7f800000, 0x7dbd57ad, 0x7dbd57ad, 0xd5048712, 0x6ddf0677, 0xedd6451f]
*)
(* @assert:metal:shape
[2, 3]
*)
(* @assert:metal:compare
cpu
*)
a * b + c - a

(* @case shape-1 *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: shaped-f32-cpu
    profile: f32-ftz
    arguments:
      - name: a
        dtype: f32
        shape: [3]
        values: [0x7dd6bdba, 0xb718b1d1, 0xbcf240fc]
      - name: b
        dtype: f32
        shape: [2, 1]
        values: [0x7d24f35f, 0x148e2932]
      - name: c
        dtype: f32
        shape: [1, 1]
        values: [0x7d3e596c]
  - name: metal
    pipeline: shaped-f32-metal
    profile: f32-ftz
    threads-per-group: 4
    arguments:
      - name: a
        dtype: f32
        shape: [3]
        values: [0x7dd6bdba, 0xb718b1d1, 0xbcf240fc]
      - name: b
        dtype: f32
        shape: [2, 1]
        values: [0x7d24f35f, 0x148e2932]
      - name: c
        dtype: f32
        shape: [1, 1]
        values: [0x7d3e596c]
*)
(* @assert:cpu:result
[0x7f800000, 0x7d3e590a, 0x7d3978ad, 0xfd6f2208, 0x7d3e596c, 0x7d3e596c]
*)
(* @assert:cpu:shape
[2, 3]
*)
(* @assert:metal:result
[0x7f800000, 0x7d3e590a, 0x7d3978ad, 0xfd6f2208, 0x7d3e596c, 0x7d3e596c]
*)
(* @assert:metal:shape
[2, 3]
*)
(* @assert:metal:compare
cpu
*)
a * b + c - a

(* @case shape-2 *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: shaped-f32-cpu
    profile: f32-ftz
    arguments:
      - name: a
        dtype: f32
        shape: [1]
        values: [0x7d57bf79]
      - name: b
        dtype: f32
        shape: [3, 1]
        values: [0x7d712586, 0x5a373f2d, 0xffce16a8]
      - name: c
        dtype: f32
        shape: [1, 5]
        values: [0x7e55bbfb, 0x2b32811e, 0x295f75e5, 0xe29e2e00, 0x42cb495f]
  - name: metal
    pipeline: shaped-f32-metal
    profile: f32-ftz
    threads-per-group: 4
    arguments:
      - name: a
        dtype: f32
        shape: [1]
        values: [0x7d57bf79]
      - name: b
        dtype: f32
        shape: [3, 1]
        values: [0x7d712586, 0x5a373f2d, 0xffce16a8]
      - name: c
        dtype: f32
        shape: [1, 5]
        values: [0x7e55bbfb, 0x2b32811e, 0x295f75e5, 0xe29e2e00, 0x42cb495f]
*)
(* @assert:cpu:result
[0x7f800000, 0x7f800000, 0x7f800000, 0x7f800000, 0x7f800000, 0x7f800000, 0x7f800000, 0x7f800000, 0x7f800000, 0x7f800000, 0x7fc00000, 0x7fc00000, 0x7fc00000, 0x7fc00000, 0x7fc00000]
*)
(* @assert:cpu:shape
[3, 5]
*)
(* @assert:metal:result
[0x7f800000, 0x7f800000, 0x7f800000, 0x7f800000, 0x7f800000, 0x7f800000, 0x7f800000, 0x7f800000, 0x7f800000, 0x7f800000, 0x7fc00000, 0x7fc00000, 0x7fc00000, 0x7fc00000, 0x7fc00000]
*)
(* @assert:metal:shape
[3, 5]
*)
(* @assert:metal:compare
cpu
*)
a * b + c - a

(* @case shape-3 *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: shaped-f32-cpu
    profile: f32-ftz
    arguments:
      - name: a
        dtype: f32
        shape: [1, 7]
        values: [0x7e6f2208, 0x426addc7, 0xd8a8807a, 0xec181591, 0xed68d1bc, 0x89ae81eb, 0x72022e4e]
      - name: b
        dtype: f32
        shape: [7]
        values: [0x7e888815, 0x59a33a70, 0x87f18b0f, 0xf591fd22, 0x98065a19, 0x49187ca4, 0xe05a9fb3]
      - name: c
        dtype: f32
        shape: [1]
        values: [0x7ea1ee22]
  - name: metal
    pipeline: shaped-f32-metal
    profile: f32-ftz
    threads-per-group: 4
    arguments:
      - name: a
        dtype: f32
        shape: [1, 7]
        values: [0x7e6f2208, 0x426addc7, 0xd8a8807a, 0xec181591, 0xed68d1bc, 0x89ae81eb, 0x72022e4e]
      - name: b
        dtype: f32
        shape: [7]
        values: [0x7e888815, 0x59a33a70, 0x87f18b0f, 0xf591fd22, 0x98065a19, 0x49187ca4, 0xe05a9fb3]
      - name: c
        dtype: f32
        shape: [1]
        values: [0x7ea1ee22]
*)
(* @assert:cpu:result
[0x7f800000, 0x7ea1ee22, 0x7ea1ee22, 0x7f800000, 0x7ea1ee22, 0x7ea1ee22, 0xff800000]
*)
(* @assert:cpu:shape
[1, 7]
*)
(* @assert:metal:result
[0x7f800000, 0x7ea1ee22, 0x7ea1ee22, 0x7f800000, 0x7ea1ee22, 0x7ea1ee22, 0xff800000]
*)
(* @assert:metal:shape
[1, 7]
*)
(* @assert:metal:compare
cpu
*)
a * b + c - a

(* @case shape-4 *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: shaped-f32-cpu
    profile: f32-ftz
    arguments:
      - name: a
        dtype: f32
        shape: [7, 1]
        values: [0x7df023c7, 0xce510e7a, 0x6c3b4b91, 0xbcb68fbc, 0x985527eb, 0xcc9c9c4e, 0x4a47f755]
      - name: b
        dtype: f32
        shape: [1]
        values: [0x7e0989d4]
      - name: c
        dtype: f32
        shape: [1, 1]
        values: [0x7e22efe1]
  - name: metal
    pipeline: shaped-f32-metal
    profile: f32-ftz
    threads-per-group: 4
    arguments:
      - name: a
        dtype: f32
        shape: [7, 1]
        values: [0x7df023c7, 0xce510e7a, 0x6c3b4b91, 0xbcb68fbc, 0x985527eb, 0xcc9c9c4e, 0x4a47f755]
      - name: b
        dtype: f32
        shape: [1]
        values: [0x7e0989d4]
      - name: c
        dtype: f32
        shape: [1, 1]
        values: [0x7e22efe1]
*)
(* @assert:cpu:result
[0x7f800000, 0xff800000, 0x7f800000, 0x7e1fdf37, 0x7e22efe1, 0xff800000, 0x7f800000]
*)
(* @assert:cpu:shape
[7, 1]
*)
(* @assert:metal:result
[0x7f800000, 0xff800000, 0x7f800000, 0x7e1fdf37, 0x7e22efe1, 0xff800000, 0x7f800000]
*)
(* @assert:metal:shape
[7, 1]
*)
(* @assert:metal:compare
cpu
*)
a * b + c - a

(* @case shape-5 *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: shaped-f32-cpu
    profile: f32-ftz
    arguments:
      - name: a
        dtype: f32
        shape: [3, 5]
        values: [0x7e3c55ee, 0x13fa2475, 0x7a166b50, 0xd924466f, 0x982dc102, 0x0ada8c79, 0x95514b84, 0x958a6113, 0x71fe7356, 0xabb512bd, 0xee8d34f8, 0xc8ed73f7, 0x5fd140ea, 0xd0c67b41, 0xf4f81bac]
      - name: b
        dtype: f32
        shape: [1, 5]
        values: [0x7f20ec63, 0xe4f56666, 0xa3a7ca8d, 0x2e6d6a88, 0x97b78c47]
      - name: c
        dtype: f32
        shape: [3, 1]
        values: [0x7f3a5270, 0xfc2dc30f, 0x52f0d522]
  - name: metal
    pipeline: shaped-f32-metal
    profile: f32-ftz
    threads-per-group: 4
    arguments:
      - name: a
        dtype: f32
        shape: [3, 5]
        values: [0x7e3c55ee, 0x13fa2475, 0x7a166b50, 0xd924466f, 0x982dc102, 0x0ada8c79, 0x95514b84, 0x958a6113, 0x71fe7356, 0xabb512bd, 0xee8d34f8, 0xc8ed73f7, 0x5fd140ea, 0xd0c67b41, 0xf4f81bac]
      - name: b
        dtype: f32
        shape: [1, 5]
        values: [0x7f20ec63, 0xe4f56666, 0xa3a7ca8d, 0x2e6d6a88, 0x97b78c47]
      - name: c
        dtype: f32
        shape: [3, 1]
        values: [0x7f3a5270, 0xfc2dc30f, 0x52f0d522]
*)
(* @assert:cpu:result
[0x7f800000, 0x7f3a5270, 0x7f3a2cd5, 0x7f3a5270, 0x7f3a5270, 0xfc2dc30f, 0xfc2dc30f, 0xfc2dc30f, 0xfc2dc317, 0xfc2dc30f, 0xff800000, 0x6e639ef6, 0xdfd140ea, 0x52fd3cd6, 0x74f81bac]
*)
(* @assert:cpu:shape
[3, 5]
*)
(* @assert:metal:result
[0x7f800000, 0x7f3a5270, 0x7f3a2cd5, 0x7f3a5270, 0x7f3a5270, 0xfc2dc30f, 0xfc2dc30f, 0xfc2dc30f, 0xfc2dc317, 0xfc2dc30f, 0xff800000, 0x6e639ef6, 0xdfd140ea, 0x52fd3cd6, 0x74f81bac]
*)
(* @assert:metal:shape
[3, 5]
*)
(* @assert:metal:compare
cpu
*)
a * b + c - a

(* @case shape-6 *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: shaped-f32-cpu
    profile: f32-ftz
    arguments:
      - name: a
        dtype: f32
        shape: [5, 3]
        values: [0x7f53b87d, 0x13661fb8, 0x0239dfb7, 0x416139aa, 0xecf29d01, 0x4468526c, 0x531e2adb, 0xa96d627e, 0x0a8927c5, 0x4d7b7660, 0xec3e363f, 0x515bce92, 0x1ea89cc9, 0x4917ff94, 0x77efe5e3]
      - name: b
        dtype: f32
        shape: [3]
        values: [0x7f6d1e8a, 0x2a9e7c61, 0xb182ea4c]
      - name: c
        dtype: f32
        shape: [5, 1]
        values: [0x7ebb542f, 0x8813f3c2, 0xe683a039, 0x0885cc44, 0xed416ad3]
  - name: metal
    pipeline: shaped-f32-metal
    profile: f32-ftz
    threads-per-group: 4
    arguments:
      - name: a
        dtype: f32
        shape: [5, 3]
        values: [0x7f53b87d, 0x13661fb8, 0x0239dfb7, 0x416139aa, 0xecf29d01, 0x4468526c, 0x531e2adb, 0xa96d627e, 0x0a8927c5, 0x4d7b7660, 0xec3e363f, 0x515bce92, 0x1ea89cc9, 0x4917ff94, 0x77efe5e3]
      - name: b
        dtype: f32
        shape: [3]
        values: [0x7f6d1e8a, 0x2a9e7c61, 0xb182ea4c]
      - name: c
        dtype: f32
        shape: [5, 1]
        values: [0x7ebb542f, 0x8813f3c2, 0xe683a039, 0x0885cc44, 0xed416ad3]
*)
(* @assert:cpu:result
[0x7f800000, 0x7ebb542f, 0x7ebb542f, 0x7f800000, 0x6cf29d01, 0xc468526c, 0x7f800000, 0xe683a039, 0xe683a039, 0x7f800000, 0x6c3e363f, 0xd15bce92, 0xed416ad3, 0xed416ad3, 0xf7efe5e9]
*)
(* @assert:cpu:shape
[5, 3]
*)
(* @assert:metal:result
[0x7f800000, 0x7ebb542f, 0x7ebb542f, 0x7f800000, 0x6cf29d01, 0xc468526c, 0x7f800000, 0xe683a039, 0xe683a039, 0x7f800000, 0x6c3e363f, 0xd15bce92, 0xed416ad3, 0xed416ad3, 0xf7efe5e9]
*)
(* @assert:metal:shape
[5, 3]
*)
(* @assert:metal:compare
cpu
*)
a * b + c - a

(* @case shape-7 *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: shaped-f32-cpu
    profile: f32-ftz
    arguments:
      - name: a
        dtype: f32
        shape: [2, 17]
        values: [0x7ed4ba3c, 0x9f4c506b, 0x95ccaace, 0x11ffb3d5, 0x97def330, 0x87566ccf, 0x2b63f3e2, 0x14c061d9, 0x7f616164, 0x03f2bd73, 0xa7796436, 0xe4cc8e1d, 0x3c46b8d8, 0x86c46657, 0x53adcfca, 0x07c6fca1, 0x95eaed8c, 0xa6aecb7b, 0x37654a9e, 0xb1bcb165, 0xc48f3380, 0x4db990df, 0x445b28b2, 0x00a8f069, 0x0f0dfeb4, 0x73349a83, 0x2275fe06, 0xce393dad, 0x89cf0328, 0x061c0c67, 0xeedb9e9a, 0x5bd15d31, 0x6afb34dc, 0xebbc4a8b]
      - name: b
        dtype: f32
        shape: [1, 17]
        values: [0x7eee2049, 0xb684ad14, 0x4515b563, 0x1b799b66, 0x427c7b8d, 0x46c06788, 0x99bc6547, 0xff4961fa, 0x019c8911, 0x12a7af3c, 0x1ba0c16b, 0x941c67ce, 0x365a4cd5, 0x7a6eb830, 0x8e216dcf, 0x7f1900e2, 0xbd200ad9]
      - name: c
        dtype: f32
        shape: [2, 1]
        values: [0x7f078656, 0xcdbd09bd]
  - name: metal
    pipeline: shaped-f32-metal
    profile: f32-ftz
    threads-per-group: 4
    arguments:
      - name: a
        dtype: f32
        shape: [2, 17]
        values: [0x7ed4ba3c, 0x9f4c506b, 0x95ccaace, 0x11ffb3d5, 0x97def330, 0x87566ccf, 0x2b63f3e2, 0x14c061d9, 0x7f616164, 0x03f2bd73, 0xa7796436, 0xe4cc8e1d, 0x3c46b8d8, 0x86c46657, 0x53adcfca, 0x07c6fca1, 0x95eaed8c, 0xa6aecb7b, 0x37654a9e, 0xb1bcb165, 0xc48f3380, 0x4db990df, 0x445b28b2, 0x00a8f069, 0x0f0dfeb4, 0x73349a83, 0x2275fe06, 0xce393dad, 0x89cf0328, 0x061c0c67, 0xeedb9e9a, 0x5bd15d31, 0x6afb34dc, 0xebbc4a8b]
      - name: b
        dtype: f32
        shape: [1, 17]
        values: [0x7eee2049, 0xb684ad14, 0x4515b563, 0x1b799b66, 0x427c7b8d, 0x46c06788, 0x99bc6547, 0xff4961fa, 0x019c8911, 0x12a7af3c, 0x1ba0c16b, 0x941c67ce, 0x365a4cd5, 0x7a6eb830, 0x8e216dcf, 0x7f1900e2, 0xbd200ad9]
      - name: c
        dtype: f32
        shape: [2, 1]
        values: [0x7f078656, 0xcdbd09bd]
*)
(* @assert:cpu:result
[0x7f800000, 0x7f078656, 0x7f078656, 0x7f078656, 0x7f078656, 0x7f078656, 0x7f078656, 0x7f078656, 0xfeb3b61c, 0x7f078656, 0x7f078656, 0x7f078656, 0x7f078656, 0x7f078656, 0x7f078656, 0x7f078656, 0x7f078656, 0xe6229738, 0xcdbd09bd, 0xcdbd09bd, 0xcdbd0999, 0x50b129c0, 0xcdb2be66, 0xcdbd09bd, 0xcf07548b, 0xf3349a83, 0xcdbd09bd, 0x4db5719d, 0xcdbd09bd, 0xcdbd09bd, 0xff800000, 0xdbd15d31, 0x7f800000, 0x6bc3a5f4]
*)
(* @assert:cpu:shape
[2, 17]
*)
(* @assert:metal:result
[0x7f800000, 0x7f078656, 0x7f078656, 0x7f078656, 0x7f078656, 0x7f078656, 0x7f078656, 0x7f078656, 0xfeb3b61c, 0x7f078656, 0x7f078656, 0x7f078656, 0x7f078656, 0x7f078656, 0x7f078656, 0x7f078656, 0x7f078656, 0xe6229738, 0xcdbd09bd, 0xcdbd09bd, 0xcdbd0999, 0x50b129c0, 0xcdb2be66, 0xcdbd09bd, 0xcf07548b, 0xf3349a83, 0xcdbd09bd, 0x4db5719d, 0xcdbd09bd, 0xcdbd09bd, 0xff800000, 0xdbd15d31, 0x7f800000, 0x6bc3a5f4]
*)
(* @assert:metal:shape
[2, 17]
*)
(* @assert:metal:compare
cpu
*)
a * b + c - a

(* @case shape-8 *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: shaped-f32-cpu
    profile: f32-ftz
    arguments:
      - name: a
        dtype: f32
        shape: [1, 6]
        values: [0x7fec1ccb, 0x9eb84bae, 0x1df01f35, 0x7a3ca710, 0xeca3cf2f, 0xc0e432c2]
      - name: b
        dtype: f32
        shape: [6]
        values: [0x800582d8, 0xb5f0a857, 0xcd3929ca, 0x83b68ea1, 0x9741578c, 0x804e2d7b]
      - name: c
        dtype: f32
        shape: [1]
        values: [0x801ee8e5]
  - name: metal
    pipeline: shaped-f32-metal
    profile: f32-ftz
    threads-per-group: 4
    arguments:
      - name: a
        dtype: f32
        shape: [1, 6]
        values: [0x7fec1ccb, 0x9eb84bae, 0x1df01f35, 0x7a3ca710, 0xeca3cf2f, 0xc0e432c2]
      - name: b
        dtype: f32
        shape: [6]
        values: [0x800582d8, 0xb5f0a857, 0xcd3929ca, 0x83b68ea1, 0x9741578c, 0x804e2d7b]
      - name: c
        dtype: f32
        shape: [1]
        values: [0x801ee8e5]
*)
(* @assert:cpu:result
[0x7fc00000, 0x1eb84bc4, 0xabadadc0, 0xfa3ca710, 0x6ca3cf2f, 0x40e432c2]
*)
(* @assert:cpu:shape
[1, 6]
*)
(* @assert:metal:result
[0x7fc00000, 0x1eb84bc4, 0xabadadc0, 0xfa3ca710, 0x6ca3cf2f, 0x40e432c2]
*)
(* @assert:metal:shape
[1, 6]
*)
(* @assert:metal:compare
cpu
*)
a * b + c - a

(* @case shape-9 *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: shaped-f32-cpu
    profile: f32-ftz
    arguments:
      - name: a
        dtype: f32
        shape: [6]
        values: [0x80384ef2, 0xe46161a9, 0x2bcb3ef4, 0x96aa5dc3, 0xec7c6846, 0xff2222ed]
      - name: b
        dtype: f32
        shape: [1, 6]
        values: [0x7f868497, 0x41d6d90a, 0x60cbf4e1, 0x545508cc, 0x422dadbb, 0xc33c47de]
      - name: c
        dtype: f32
        shape: [1, 1]
        values: [0x7f9feaa4]
  - name: metal
    pipeline: shaped-f32-metal
    profile: f32-ftz
    threads-per-group: 4
    arguments:
      - name: a
        dtype: f32
        shape: [6]
        values: [0x80384ef2, 0xe46161a9, 0x2bcb3ef4, 0x96aa5dc3, 0xec7c6846, 0xff2222ed]
      - name: b
        dtype: f32
        shape: [1, 6]
        values: [0x7f868497, 0x41d6d90a, 0x60cbf4e1, 0x545508cc, 0x422dadbb, 0xc33c47de]
      - name: c
        dtype: f32
        shape: [1, 1]
        values: [0x7f9feaa4]
*)
(* @assert:cpu:result
[0x7fc00000, 0x7fc00000, 0x7fc00000, 0x7fc00000, 0x7fc00000, 0x7fc00000]
*)
(* @assert:cpu:shape
[1, 6]
*)
(* @assert:metal:result
[0x7fc00000, 0x7fc00000, 0x7fc00000, 0x7fc00000, 0x7fc00000, 0x7fc00000]
*)
(* @assert:metal:shape
[1, 6]
*)
(* @assert:metal:compare
cpu
*)
a * b + c - a

(* @case shape-10 *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: shaped-f32-cpu
    profile: f32-ftz
    arguments:
      - name: a
        dtype: f32
        shape: [1, 1]
        values: [0x7fb950b1]
      - name: b
        dtype: f32
        shape: [1]
        values: [0x7fd2b6be]
      - name: c
        dtype: f32
        shape: [1, 1]
        values: [0x80b74d33]
  - name: metal
    pipeline: shaped-f32-metal
    profile: f32-ftz
    threads-per-group: 4
    arguments:
      - name: a
        dtype: f32
        shape: [1, 1]
        values: [0x7fb950b1]
      - name: b
        dtype: f32
        shape: [1]
        values: [0x7fd2b6be]
      - name: c
        dtype: f32
        shape: [1, 1]
        values: [0x80b74d33]
*)
(* @assert:cpu:result
[0x7fc00000]
*)
(* @assert:cpu:shape
[1, 1]
*)
(* @assert:metal:result
[0x7fc00000]
*)
(* @assert:metal:shape
[1, 1]
*)
(* @assert:metal:compare
cpu
*)
a * b + c - a

(* @case shape-11 *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: shaped-f32-cpu
    profile: f32-ftz
    arguments:
      - name: a
        dtype: f32
        shape: [2, 3]
        values: [0x80d0b340, 0x6fb38d9f, 0x47817e72, 0xcf85cb29, 0xec2d9a74, 0x7b9e0343]
      - name: b
        dtype: f32
        shape: [2, 3]
        values: [0x80ea194d, 0x86ebea48, 0xf6ca8907, 0xd8ffb2ba, 0x96cb22d1, 0x3b07fdfc]
      - name: c
        dtype: f32
        shape: [2, 3]
        values: [0x81037f5a, 0x9e2446f1, 0xa613939c, 0xe2799a4b, 0x4168ab2e, 0xfa71f8b5]
  - name: metal
    pipeline: shaped-f32-metal
    profile: f32-ftz
    threads-per-group: 4
    arguments:
      - name: a
        dtype: f32
        shape: [2, 3]
        values: [0x80d0b340, 0x6fb38d9f, 0x47817e72, 0xcf85cb29, 0xec2d9a74, 0x7b9e0343]
      - name: b
        dtype: f32
        shape: [2, 3]
        values: [0x80ea194d, 0x86ebea48, 0xf6ca8907, 0xd8ffb2ba, 0x96cb22d1, 0x3b07fdfc]
      - name: c
        dtype: f32
        shape: [2, 3]
        values: [0x81037f5a, 0x9e2446f1, 0xa613939c, 0xe2799a4b, 0x4168ab2e, 0xfa71f8b5]
*)
(* @assert:cpu:result
[0x80000000, 0xefb38d9f, 0xfecce62c, 0x69059ee0, 0x6c2d9a74, 0xfbbbee69]
*)
(* @assert:cpu:shape
[2, 3]
*)
(* @assert:metal:result
[0x80000000, 0xefb38d9f, 0xfecce62c, 0x69059ee0, 0x6c2d9a74, 0xfbbbee69]
*)
(* @assert:metal:shape
[2, 3]
*)
(* @assert:metal:compare
cpu
*)
a * b + c - a

(* @case unused-input *)
(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: shaped-f32-cpu
    profile: f32-ftz
    arguments:
      - name: a
        dtype: f32
        shape: [3]
        values: [0x3f000000, 0x3fc00000, 0xbf000000]
      - name: b
        dtype: f32
        shape: [2, 1]
        values: [0x7eee2049, 0xb684ad14]
  - name: metal
    pipeline: shaped-f32-metal
    profile: f32-ftz
    arguments:
      - name: a
        dtype: f32
        shape: [3]
        values: [0x3f000000, 0x3fc00000, 0xbf000000]
      - name: b
        dtype: f32
        shape: [2, 1]
        values: [0x7eee2049, 0xb684ad14]
*)
(* @assert:cpu:result
[0x3f400000, 0x3fe00000, 0xbe800000, 0x3f400000, 0x3fe00000, 0xbe800000]
*)
(* @assert:cpu:shape
[2, 3]
*)
(* @assert:metal:result
[0x3f400000, 0x3fe00000, 0xbe800000, 0x3f400000, 0x3fe00000, 0xbe800000]
*)
(* @assert:metal:shape
[2, 3]
*)
(* @assert:metal:compare
cpu
*)
a + 0.25

