# Tutorial: Ethereum (Foundry) — Build, Test, and Deploy from ARTHEN Output

Goal: Gunakan ARTHEN untuk menghasilkan Solidity, kompilasi stub kontrak menggunakan Foundry (forge), jalankan tes, dan siapkan deploy.

Prerequisites
- Foundry (forge, cast, anvil)
- Node.js (opsional untuk skrip tambahan)
- Python 3.10+ dengan pip

Install Foundry
- macOS/Linux: `curl -L https://foundry.paradigm.xyz | bash && foundryup`
- Windows: gunakan `foundryup` via PowerShell (lihat dokumentasi Foundry)

1) Generate Solidity dari ARTHEN
```bash
pip install -e .
python scripts/generate_ethereum_solidity.py
# artefak: build_artifacts/ethereum/ai_governance_system.sol
```

2) Inisialisasi proyek Foundry
```bash
mkdir -p examples/ethereum/foundry-minimal
cd examples/ethereum/foundry-minimal
forge init --no-interactive
mkdir -p src test script
copy ..\..\..\build_artifacts\ethereum\ai_governance_system.sol src\ArthenCompiledStub.sol
```

3) Tambahkan test (test/ArthenCompiledStub.t.sol)
```solidity
// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.21;
import "forge-std/Test.sol";
import "../src/ArthenCompiledStub.sol";
contract ArthenCompiledStubTest is Test {
    ArthenCompiledStub stub;
    function setUp() public { stub = new ArthenCompiledStub(); }
    function testPing() public {
        uint256 res = stub.ping(1);
        assertEq(res, 43);
    }
}
```

4) Build & Test
```bash
forge build
forge test -vv
```

5) Deploy skrip (script/Deploy.s.sol)
```solidity
// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.21;
import "forge-std/Script.sol";
import "../src/ArthenCompiledStub.sol";
contract DeployScript is Script {
    function run() external {
        uint256 pk = vm.envUint("PRIVATE_KEY");
        vm.startBroadcast(pk);
        ArthenCompiledStub c = new ArthenCompiledStub();
        vm.stopBroadcast();
    }
}
```

Jalankan deploy (contoh lokal/anvil)
```bash
anvil &
export PRIVATE_KEY=<hex_private_key>
forge script script/Deploy.s.sol --rpc-url http://localhost:8545 --broadcast
```

Catatan
- Artefak ARTHEN adalah stub kontrak yang valid untuk kompilasi CI. Adaptasi ke proyek Anda untuk produksi.
- Gunakan `solhint` dan `slither` di luar Foundry untuk lint/analisis keamanan tambahan.
- Lihat docs/HARDENING_GUIDE.md untuk praktik keamanan.