# Tutorial: Ethereum (Sepolia) — Build & Deploy from ARTHEN Output

Goal: Use ARTHEN to generate Solidity, compile a minimal stub, and prepare for deployment to Sepolia.

Prerequisites
- Node.js 18+
- Python 3.10+ with pip
- git
- A Sepolia RPC endpoint and funded account (for deployment)

1) Install ARTHEN and generate Solidity
```bash
# in repository root
pip install -e .
python scripts/generate_ethereum_solidity.py
# artifact: build_artifacts/ethereum/ai_governance_system.sol
```

2) Quick compile check with solc (optional)
```bash
python -c "from solcx import install_solc,set_solc_version,compile_files; install_solc('0.8.21'); set_solc_version('0.8.21'); from pathlib import Path; res=compile_files([str(Path('build_artifacts/ethereum/ai_governance_system.sol'))]); print('solc compile OK:', bool(res))"
```

3) Hardhat minimal project
```bash
mkdir -p examples/ethereum/hardhat-minimal
cd examples/ethereum/hardhat-minimal
npm init -y
npm i --save-dev hardhat @nomicfoundation/hardhat-toolbox dotenv
npx hardhat init --force
mkdir -p contracts
copy ..\..\..\build_artifacts\ethereum\ai_governance_system.sol contracts\ArthenCompiledStub.sol
```

Add a deploy script (scripts/deploy.ts)
```ts
import { ethers } from "hardhat";
async function main() {
  const Factory = await ethers.getContractFactory("ArthenCompiledStub");
  const contract = await Factory.deploy();
  await contract.deployed();
  console.log("Deployed:", contract.address);
}
main().catch((e) => { console.error(e); process.exit(1); });
```

Configure network (hardhat.config.ts)
```ts
import { HardhatUserConfig } from "hardhat/config";
import "@nomicfoundation/hardhat-toolbox";
import dotenv from "dotenv";dotenv.config();
const config: HardhatUserConfig = {
  solidity: "0.8.21",
  networks: {
    sepolia: {
      url: process.env.SEPOLIA_RPC!,
      accounts: [process.env.SEPOLIA_PRIVATE_KEY!],
    },
  },
};
export default config;
```

Compile & deploy
```bash
npx hardhat compile
npx hardhat run scripts/deploy.ts --network sepolia
```

Security lint (best-effort)
```bash
npx solhint contracts/ArthenCompiledStub.sol || true
```

Notes
- The generated Solidity is embedded as a doc-comment plus a minimal stub to ensure CI compile success. For production-grade use, adapt the emitted code to your project and ensure all dependencies exist.
- See docs/HARDENING_GUIDE.md for security practices.