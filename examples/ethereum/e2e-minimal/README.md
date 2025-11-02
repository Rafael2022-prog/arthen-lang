# Ethereum E2E Minimal (Draft)

This example shows how to adapt ARTHEN compiled Solidity into a minimal Hardhat/Foundry project for deployment to Sepolia.

Status: Draft. You may need to adjust the contract and tool versions.

## Steps (Hardhat)
1. Create project:
   ```bash
   mkdir -p contracts scripts
   npm init -y
   npm i --save-dev hardhat @nomicfoundation/hardhat-toolbox
   npx hardhat
   ```
2. Copy ARTHEN output:
   - Generate Solidity using `python scripts/generate_ethereum_solidity.py`
   - Copy `build_artifacts/ethereum/ai_governance_system.sol` to `contracts/ArthenOutput.sol`
3. Configure network (Sepolia) in `hardhat.config.js` with your RPC and private key.
4. Deploy:
   ```bash
   npx hardhat run scripts/deploy.js --network sepolia
   ```

## Sample deploy script
Create `scripts/deploy.js`:
```js
async function main() {
  const fs = require('fs');
  const path = require('path');
  const sourcePath = path.join(__dirname, '..', 'contracts', 'ArthenOutput.sol');
  console.log('Using contract source:', sourcePath);
  // Standard Hardhat compile & deploy flow should be placed here.
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
```

## Notes
- Use `solhint` and `slither` as per docs/HARDENING_GUIDE.md.
- Consider Foundry setup if preferred.