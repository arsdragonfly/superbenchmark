target triple = "amdgcn-amd-amdhsa"

declare i32 @llvm.amdgcn.workitem.id.x()

define amdgpu_kernel void @read_u32(ptr addrspace(1) %out, ptr addrspace(1) %src, i32 %offset) {
entry:
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  %index = add i32 %id, %offset
  %wide = zext i32 %index to i64
  %address = getelementptr i32, ptr addrspace(1) %src, i64 %wide
  %value = load i32, ptr addrspace(1) %address, align 4
  %output = getelementptr i32, ptr addrspace(1) %out, i32 %id
  store i32 %value, ptr addrspace(1) %output, align 4
  ret void
}
