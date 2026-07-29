# Req1 Environment Audit

- status: `pass`
- root: `/home/kjm26/project/PRJXR/XR-VIT/HGTXR`
- hardware_root: `/home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware`
- checks: `13/13`
- fail_count: `0`

## Environment

- platform_system: `Linux`
- platform_release: `6.8.0-124-generic`
- os_release: `{'PRETTY_NAME': 'Ubuntu 22.04.5 LTS', 'NAME': 'Ubuntu', 'VERSION_ID': '22.04', 'VERSION': '22.04.5 LTS (Jammy Jellyfish)', 'VERSION_CODENAME': 'jammy', 'ID': 'ubuntu', 'ID_LIKE': 'debian', 'HOME_URL': 'https://www.ubuntu.com/', 'SUPPORT_URL': 'https://help.ubuntu.com/', 'BUG_REPORT_URL': 'https://bugs.launchpad.net/ubuntu/', 'PRIVACY_POLICY_URL': 'https://www.ubuntu.com/legal/terms-and-policies/privacy-policy', 'UBUNTU_CODENAME': 'jammy'}`
- proc_version: `Linux version 6.8.0-124-generic (buildd@lcy02-amd64-109) (x86_64-linux-gnu-gcc-12 (Ubuntu 12.3.0-1ubuntu1~22.04.3) 12.3.0, GNU ld (GNU Binutils for Ubuntu) 2.38) #124~22.04.1-Ubuntu SMP PREEMPT_DYNAMIC Tue May 26 21:05:19 UTC `
- vitis_hls: `/tools/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls`
- vivado: `/tools/Xilinx/Vivado/2023.2/bin/vivado`

## Checks

| Check | Status | Detail |
|---|---|---|
| platform_linux | `pass` | Linux |
| ubuntu_id | `pass` | {'PRETTY_NAME': 'Ubuntu 22.04.5 LTS', 'NAME': 'Ubuntu', 'VERSION_ID': '22.04', 'VERSION': '22.04.5 LTS (Jammy Jellyfish)', 'VERSION_CODENAME': 'jammy', 'ID': 'ubuntu', 'ID_LIKE': 'debian', 'HOME_URL': 'https://www.ubuntu.com/', 'SUPPORT_URL': 'https://help.ubuntu.com/', 'BUG_REPORT_URL': 'https://bugs.launchpad.net/ubuntu/', 'PRIVACY_POLICY_URL': 'https://www.ubuntu.com/legal/terms-and-policies/privacy-policy', 'UBUNTU_CODENAME': 'jammy'} |
| ubuntu_version_22_04 | `pass` | 22.04 |
| not_wsl_kernel | `pass` | Linux version 6.8.0-124-generic (buildd@lcy02-amd64-109) (x86_64-linux-gnu-gcc-12 (Ubuntu 12.3.0-1ubuntu1~22.04.3) 12.3.0, GNU ld (GNU Binutils for Ubuntu) 2.38) #124~22.04.1-Ubuntu SMP PREEMPT_DYNAMIC Tue May 26 21:05:19 UTC  |
| hgtxr_root_expected_suffix | `pass` | /home/kjm26/project/PRJXR/XR-VIT/HGTXR |
| hardware_root_expected_suffix | `pass` | /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware |
| hardware_root_expected_prefix | `pass` | /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware |
| hardware_root_exists | `pass` | /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware |
| no_legacy_wsl_or_xilinx_paths | `pass` | [] |
| vitis_hls_tools_xilinx_exists | `pass` | /tools/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls |
| vitis_hls_tools_xilinx_executable | `pass` | /tools/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls |
| vivado_tools_xilinx_exists | `pass` | /tools/Xilinx/Vivado/2023.2/bin/vivado |
| vivado_tools_xilinx_executable | `pass` | /tools/Xilinx/Vivado/2023.2/bin/vivado |

## Safety

- executes_tools: `False`
- executes_hls: `False`
- executes_vivado: `False`
- writes_canonical_inputs: `False`
