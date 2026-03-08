// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "forge-std/Test.sol";
import "../src/TransplantManagement.sol";

/**
 * @title TransplantManagementV21Test
 * @dev Test suite for TransplantManagement v2.1 (primary + backup + matchCID + backup promotion).
 *
 * IMPORTANT:
 * - This test assumes the new contract has:
 *   createMatch(donorId, primaryRecipientId, backupRecipientId, matchCID)
 *   promoteBackupRecipient(matchId)
 *   approveRecipient(matchId) => only active recipient (primary by default)
 *
 * - If your revert strings differ slightly, update vm.expectRevert(...) accordingly.
 */
contract TransplantManagementV21Test is Test {
    TransplantManagement transplant;

    // Actors
    address regulator = address(0x1);
    address hospital = address(0x2);
    address medicalTeam = address(0x3);
    address llm = address(0x4);
    address ethicalCommittee = address(0x5);

    address donorAddr = address(0x6);
    address recipient1Addr = address(0x7);
    address recipient2Addr = address(0x8);
    address unauthorized = address(0x9);

    // Simple CIDs
    string donorCID = "bafyDonorCID";
    string r1CID = "bafyRecipient1CID";
    string r2CID = "bafyRecipient2CID";
    string matchCID = "bafyMatchRationaleCID";

    function setUp() public {
        // Deploy contract with regulator
        vm.prank(regulator);
        transplant = new TransplantManagement(regulator);

        // Register entities
        vm.prank(regulator);
        transplant.registerHospital(hospital);

        vm.prank(regulator);
        transplant.registerMedicalTeam(medicalTeam);

        vm.prank(regulator);
        transplant.registerLLM(llm);

        vm.prank(regulator);
        transplant.registerEthicalCommittee(ethicalCommittee);

        // Pre-register donor + recipients addresses (Regulator)
        vm.prank(regulator);
        transplant.registerDonorAddress(donorAddr);

        vm.prank(regulator);
        transplant.registerRecipientAddress(recipient1Addr);

        vm.prank(regulator);
        transplant.registerRecipientAddress(recipient2Addr);
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Helpers
    // ─────────────────────────────────────────────────────────────────────────────
    function _registerDonorAndRecipients() internal {
        vm.prank(hospital);
        transplant.registerDonor(donorAddr, "O", "A1,A2,B7,DR15", "Kidney", donorCID);

        vm.prank(hospital);
        transplant.registerRecipient(recipient1Addr, "O", "A1,A3,B7,DR15", "Kidney", r1CID);

        vm.prank(hospital);
        transplant.registerRecipient(recipient2Addr, "O", "A1,A2,B8,DR4", "Kidney", r2CID);
    }

    function _ethicallyApproveAll() internal {
        // donorId = 1, recipientIds = 1 and 2 (because pre-registration increments counters)
        vm.prank(ethicalCommittee);
        transplant.approveDonorEthicalCommittee(1);

        vm.prank(ethicalCommittee);
        transplant.approveRecipientEthicalCommittee(1);

        vm.prank(ethicalCommittee);
        transplant.approveRecipientEthicalCommittee(2);
    }

    function _createMatchPrimaryBackup() internal {
        vm.prank(llm);
        transplant.createMatch(1, 1, 2, matchCID);
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Registration tests
    // ─────────────────────────────────────────────────────────────────────────────
    function testDonorAndRecipientRegistration_Success() public {
        vm.prank(hospital);
        transplant.registerDonor(donorAddr, "A+", "HLA-A2", "Kidney", donorCID);

        (uint256 donorId, address dAddr, string memory bloodType, , string memory organType, string memory ipfsHash, bool registered, bool ethicalApproved)
            = transplant.donors(1);

        assertEq(donorId, 1);
        assertEq(dAddr, donorAddr);
        assertEq(bloodType, "A+");
        assertEq(organType, "Kidney");
        assertEq(ipfsHash, donorCID);
        assertTrue(registered);
        assertFalse(ethicalApproved);

        vm.prank(hospital);
        transplant.registerRecipient(recipient1Addr, "A+", "HLA-A2", "Kidney", r1CID);

        (uint256 rid, address rAddr, string memory rBloodType, , string memory rOrganType, string memory rIpfs, bool rRegistered, bool matched, bool rEthicalApproved)
            = transplant.recipients(1);

        assertEq(rid, 1);
        assertEq(rAddr, recipient1Addr);
        assertEq(rBloodType, "A+");
        assertEq(rOrganType, "Kidney");
        assertEq(rIpfs, r1CID);
        assertTrue(rRegistered);
        assertFalse(matched);
        assertFalse(rEthicalApproved);
    }

    function testDonorRegistration_FailsForUnauthorizedCaller() public {
        // New contract revert string is "Only hospital"
        vm.expectRevert(bytes("Only hospital"));
        vm.prank(unauthorized);
        transplant.registerDonor(donorAddr, "A+", "HLA-A2", "Kidney", donorCID);
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Ethical approval tests
    // ─────────────────────────────────────────────────────────────────────────────
    function testEthicalApproval_Success() public {
        _registerDonorAndRecipients();

        vm.prank(ethicalCommittee);
        transplant.approveDonorEthicalCommittee(1);

        (, , , , , , , bool ethicalApproved) = transplant.donors(1);
        assertTrue(ethicalApproved);

        vm.prank(ethicalCommittee);
        transplant.approveRecipientEthicalCommittee(1);

        (, , , , , , , , bool rEthicalApproved) = transplant.recipients(1);
        assertTrue(rEthicalApproved);
    }

    function testEthicalApproval_FailsForUnauthorizedCaller() public {
        vm.expectRevert(bytes("Only ethical committee"));
        vm.prank(unauthorized);
        transplant.approveDonorEthicalCommittee(1);
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Match creation tests
    // ─────────────────────────────────────────────────────────────────────────────
    function testMatchCreation_Success() public {
        _registerDonorAndRecipients();
        _ethicallyApproveAll();

        _createMatchPrimaryBackup();

        // matchId = 1
        (
            uint256 matchId,
            uint256 donorId,
            uint256 primaryId,
            uint256 backupId,
            uint256 activeId,
            bool backupPromoted,
            address matchedByLLM,
            string memory storedMatchCID,
            bool medicalApproved,
            bool hospitalApproved,
            bool donorApproved,
            bool activeRecipientApproved,
            bool ethicalCommitteeApproved,
            bool finalized
        ) = transplant.matches(1);

        assertEq(matchId, 1);
        assertEq(donorId, 1);
        assertEq(primaryId, 1);
        assertEq(backupId, 2);
        assertEq(activeId, 1);
        assertFalse(backupPromoted);
        assertEq(matchedByLLM, llm);
        assertEq(storedMatchCID, matchCID);

        assertFalse(medicalApproved);
        assertFalse(hospitalApproved);
        assertFalse(donorApproved);
        assertFalse(activeRecipientApproved);
        assertFalse(ethicalCommitteeApproved);
        assertFalse(finalized);
    }

    function testMatchCreation_FailsForUnauthorizedCaller() public {
        vm.expectRevert(bytes("Only authorized LLM"));
        vm.prank(unauthorized);
        transplant.createMatch(1, 1, 2, matchCID);
    }

    function testMatchCreation_FailsIfNotEthicallyApproved() public {
        _registerDonorAndRecipients();

        // donor + recipients not ethically approved yet
        vm.expectRevert(bytes("Donor must be ethically approved"));
        vm.prank(llm);
        transplant.createMatch(1, 1, 2, matchCID);
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Full approval flow (primary recipient)
    // ─────────────────────────────────────────────────────────────────────────────
    function testFullApprovalProcess_Success_Primary() public {
        _registerDonorAndRecipients();
        _ethicallyApproveAll();
        _createMatchPrimaryBackup();

        vm.prank(medicalTeam);
        transplant.approveMedicalTeam(1);

        vm.prank(hospital);
        transplant.approveHospital(1);

        vm.prank(donorAddr);
        transplant.approveDonor(1);

        // Primary recipient approves (active recipient starts as primary = recipientId 1 => recipient1Addr)
        vm.prank(recipient1Addr);
        transplant.approveRecipient(1);

        vm.prank(ethicalCommittee);
        transplant.approveFinalTransplant(1);

        assertTrue(transplant.isTransplantApproved(1));
    }

    function testTransplantApproval_FailsIfNotFullyApproved() public {
        _registerDonorAndRecipients();
        _ethicallyApproveAll();
        _createMatchPrimaryBackup();

        vm.prank(medicalTeam);
        transplant.approveMedicalTeam(1);

        // Missing hospital + donor + recipient + ethical final approvals
        assertFalse(transplant.isTransplantApproved(1));
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Backup promotion tests
    // ─────────────────────────────────────────────────────────────────────────────
    function testBackupPromotion_Success_AndRecipientApprovalSwitches() public {
        _registerDonorAndRecipients();
        _ethicallyApproveAll();
        _createMatchPrimaryBackup();

        // Promote backup (allowed by hospital OR medical team)
        vm.prank(hospital);
        transplant.promoteBackupRecipient(1);

        // Confirm active recipient is now backup (recipientId2 => recipient2Addr)
        (
            ,
            ,
            ,
            uint256 backupId,
            uint256 activeId,
            bool backupPromoted,
            ,
            ,
            ,
            ,
            ,
            bool activeRecipientApproved,
            ,
        ) = transplant.matches(1);

        assertEq(backupId, 2);
        assertEq(activeId, 2);
        assertTrue(backupPromoted);
        assertFalse(activeRecipientApproved); // reset on promotion

        // Old primary recipient should NOT be able to approve anymore
        vm.expectRevert(bytes("Only active recipient can approve"));
        vm.prank(recipient1Addr);
        transplant.approveRecipient(1);

        // Backup recipient approves successfully
        vm.prank(recipient2Addr);
        transplant.approveRecipient(1);

        // Complete the rest of approvals
        vm.prank(medicalTeam);
        transplant.approveMedicalTeam(1);

        vm.prank(hospital);
        transplant.approveHospital(1);

        vm.prank(donorAddr);
        transplant.approveDonor(1);

        vm.prank(ethicalCommittee);
        transplant.approveFinalTransplant(1);

        assertTrue(transplant.isTransplantApproved(1));
    }

    function testBackupPromotion_FailsIfCalledTwice() public {
        _registerDonorAndRecipients();
        _ethicallyApproveAll();
        _createMatchPrimaryBackup();

        vm.prank(medicalTeam);
        transplant.promoteBackupRecipient(1);

        vm.expectRevert(bytes("Backup already promoted"));
        vm.prank(medicalTeam);
        transplant.promoteBackupRecipient(1);
    }
}