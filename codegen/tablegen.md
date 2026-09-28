## install

```
  sudo apt update
  apt-cache policy llvm-20-dev
  sudo apt install llvm-20-dev llvm-20-tools
```

```
  cd LLVM-Code-Generation/ch6
  cmake -G Ninja -DCMAKE_BUILD_TYPE=Debug \
    -DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm \
    -B build .
  ninja -C build
```

## 
https://llvm.org/docs/TableGen/ProgRef.html