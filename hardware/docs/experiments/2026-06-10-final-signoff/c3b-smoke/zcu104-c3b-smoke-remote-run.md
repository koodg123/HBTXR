# HGTXR ZCU104 Remote Smoke Runner

- status: `dry-run`
- profile: `c3b-mem16`
- preset: `axis-c3b-mem16`
- execute: `False`
- host: `zcu104.local`
- user: `xilinx`
- remote_dir: `/home/xilinx/hgtxr_c3b_smoke`
- local_result: `/home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/generated/pynq/e2e_axis_dma_c3b_mem16_file_smoke.remote.json`
- canonical_result: `/home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json`

## Commands

### mkdir
```sh
ssh xilinx@zcu104.local 'mkdir -p /home/xilinx/hgtxr_c3b_smoke'
```

### copy_tar
```sh
scp /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/generated/pynq/e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz /home/kjm26/project/PRJXR/XR-VIT/HGTXR/docs/resources/e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz.sha256 xilinx@zcu104.local:/home/xilinx/hgtxr_c3b_smoke/
```

### run
```sh
ssh xilinx@zcu104.local 'cd /home/xilinx/hgtxr_c3b_smoke && sha256sum -c e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz.sha256 && tar -xzf e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz && cd e2e_axis_dma_c3b_mem16_smoke_bundle && ./run_e2e_axis_dma_c3b_mem16_file_smoke.sh && ./validate_e2e_axis_dma_c3b_mem16_file_smoke.sh'
```

### fetch_result
```sh
scp xilinx@zcu104.local:/home/xilinx/hgtxr_c3b_smoke/e2e_axis_dma_c3b_mem16_smoke_bundle/e2e_axis_dma_c3b_mem16_file_smoke.json /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/generated/pynq/e2e_axis_dma_c3b_mem16_file_smoke.remote.json
```

### fetch_validation
```sh
scp xilinx@zcu104.local:/home/xilinx/hgtxr_c3b_smoke/e2e_axis_dma_c3b_mem16_smoke_bundle/e2e_axis_dma_c3b_mem16_file_smoke_validation.json /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/generated/pynq/e2e_axis_dma_c3b_mem16_file_smoke_validation.json
```

## Errors

- None.
