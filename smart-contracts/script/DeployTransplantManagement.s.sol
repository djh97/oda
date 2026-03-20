// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "forge-std/Script.sol";
import "forge-std/console2.sol";
import "../src/TransplantManagement.sol";

contract DeployTransplantManagement is Script {
    function run() external returns (TransplantManagement deployed) {
        address regulator = vm.envAddress("REGULATOR_ADDRESS");

        vm.startBroadcast();
        deployed = new TransplantManagement(regulator);
        vm.stopBroadcast();

        console2.log("TransplantManagement deployed at:", address(deployed));
    }
}
