// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "forge-std/Test.sol";
import "../src/TransplantManagement.sol";

/**
 * @title TransplantManagementV22Test
 * @dev Test suite for TransplantManagement v2.2
 *
 * Coverage:
 * - role/entity registration
 * - donor/recipient registration
 * - ethical eligibility approvals
 * - match creation with primary + backup + matchCID
 * - strict sequential approval flow
 * - finalizeMatch requires full approval
 * - no approvals after finalization
 * - donor open-match guard
 * - recipient matched guard
 * - backup promotion before recipient/ethics final approval only
 */
contract TransplantManagementV22Test is Test {
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
    address recipient3Addr = address(0x10);
    address unauthorized = address(0x9);

    // Simple CIDs
    string donorCID = "bafyDonorCID";
    string r1CID = "bafyRecipient1CID";
    string r2CID = "bafyRecipient2CID";
    string r3CID = "bafyRecipient3CID";
    string matchCID = "bafyMatchRationaleCID";

    function setUp() public {
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

        // Pre-register donor + recipients
        vm.prank(regulator);
        transplant.registerDonorAddress(donorAddr);

        vm.prank(regulator);
        transplant.registerRecipientAddress(recipient1Addr);

        vm.prank(regulator);
        transplant.registerRecipientAddress(recipient2Addr);

        vm.prank(regulator);
        transplant.registerRecipientAddress(recipient3Addr);
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Helpers
    // ─────────────────────────────────────────────────────────────────────────────

    function _registerDonorAndRecipients12() internal {
        vm.prank(hospital);
        transplant.registerDonor(donorAddr, "O", "A1,A2,B7,DR15", "Kidney", donorCID);

        vm.prank(hospital);
        transplant.registerRecipient(recipient1Addr, "O", "A1,A3,B7,DR15", "Kidney", r1CID);

        vm.prank(hospital);
        transplant.registerRecipient(recipient2Addr, "O", "A1,A2,B8,DR4", "Kidney", r2CID);
    }

    function _registerRecipient3() internal {
        vm.prank(hospital);
        transplant.registerRecipient(recipient3Addr, "O", "A2,A24,B44,DR4", "Kidney", r3CID);
    }

    function _ethicallyApproveDonorAndRecipients12() internal {
        vm.prank(ethicalCommittee);
        transplant.approveDonorEthicalCommittee(1);

        vm.prank(ethicalCommittee);
        transplant.approveRecipientEthicalCommittee(1);

        vm.prank(ethicalCommittee);
        transplant.approveRecipientEthicalCommittee(2);
    }

    function _ethicallyApproveRecipient3() internal {
        vm.prank(ethicalCommittee);
        transplant.approveRecipientEthicalCommittee(3);
    }

    function _createMatchPrimaryBackup12() internal {
        vm.prank(llm);
        transplant.createMatch(1, 1, 2, matchCID);
    }

    function _completeApprovalsPrimary() internal {
        vm.prank(medicalTeam);
        transplant.approveMedicalTeam(1);

        vm.prank(hospital);
        transplant.approveHospital(1);

        vm.prank(donorAddr);
        transplant.approveDonor(1);

        vm.prank(recipient1Addr);
        transplant.approveRecipient(1);

        vm.prank(ethicalCommittee);
        transplant.approveFinalTransplant(1);
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Registration tests
    // ─────────────────────────────────────────────────────────────────────────────

    function testDonorAndRecipientRegistration_Success() public {
        vm.prank(hospital);
        transplant.registerDonor(donorAddr, "A+", "HLA-A2", "Kidney", donorCID);

        (
            uint256 donorId,
            address dAddr,
            string memory bloodType,
            ,
            string memory organType,
            string memory ipfsHash,
            bool registered,
            bool ethicalApproved
        ) = transplant.donors(1);

        assertEq(donorId, 1);
        assertEq(dAddr, donorAddr);
        assertEq(bloodType, "A+");
        assertEq(organType, "Kidney");
        assertEq(ipfsHash, donorCID);
        assertTrue(registered);
        assertFalse(ethicalApproved);

        vm.prank(hospital);
        transplant.registerRecipient(recipient1Addr, "A+", "HLA-A2", "Kidney", r1CID);

        (
            uint256 rid,
            address rAddr,
            string memory rBloodType,
            ,
            string memory rOrganType,
            string memory rIpfs,
            bool rRegistered,
            bool matched,
            bool rEthicalApproved
        ) = transplant.recipients(1);

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
        vm.expectRevert(bytes("Only hospital"));
        vm.prank(unauthorized);
        transplant.registerDonor(donorAddr, "A+", "HLA-A2", "Kidney", donorCID);
    }

    function testRecipientRegistration_FailsIfEmptyCID() public {
        vm.expectRevert(bytes("Empty recipient CID"));
        vm.prank(hospital);
        transplant.registerRecipient(recipient1Addr, "A+", "HLA-A2", "Kidney", "");
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Ethical approval tests
    // ─────────────────────────────────────────────────────────────────────────────

    function testEthicalApproval_Success() public {
        _registerDonorAndRecipients12();

        vm.prank(ethicalCommittee);
        transplant.approveDonorEthicalCommittee(1);

        (, , , , , , , bool donorApproved) = transplant.donors(1);
        assertTrue(donorApproved);

        vm.prank(ethicalCommittee);
        transplant.approveRecipientEthicalCommittee(1);

        (, , , , , , , , bool recipientApproved) = transplant.recipients(1);
        assertTrue(recipientApproved);
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
        _registerDonorAndRecipients12();
        _ethicallyApproveDonorAndRecipients12();

        _createMatchPrimaryBackup12();

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

        assertTrue(transplant.donorHasOpenMatch(1));

        (, , , , , , , bool r1Matched, ) = transplant.recipients(1);
        (, , , , , , , bool r2Matched, ) = transplant.recipients(2);
        assertTrue(r1Matched);
        assertTrue(r2Matched);
    }

    function testMatchCreation_FailsForUnauthorizedCaller() public {
        vm.expectRevert(bytes("Only authorized LLM"));
        vm.prank(unauthorized);
        transplant.createMatch(1, 1, 2, matchCID);
    }

    function testMatchCreation_FailsIfNotEthicallyApproved() public {
        _registerDonorAndRecipients12();

        vm.expectRevert(bytes("Donor must be ethically approved"));
        vm.prank(llm);
        transplant.createMatch(1, 1, 2, matchCID);
    }

    function testMatchCreation_FailsIfDonorAlreadyHasOpenMatch() public {
        _registerDonorAndRecipients12();
        _registerRecipient3();
        _ethicallyApproveDonorAndRecipients12();
        _ethicallyApproveRecipient3();

        vm.prank(llm);
        transplant.createMatch(1, 1, 2, matchCID);

        vm.expectRevert(bytes("Donor already has open match"));
        vm.prank(llm);
        transplant.createMatch(1, 3, 2, "bafyAnotherMatchCID");
    }

    function testMatchCreation_FailsIfRecipientAlreadyMatched() public {
        _registerDonorAndRecipients12();
        _registerRecipient3();
        _ethicallyApproveDonorAndRecipients12();
        _ethicallyApproveRecipient3();

        vm.prank(llm);
        transplant.createMatch(1, 1, 2, matchCID);

        // Use a fresh donor address to isolate the recipient-matched guard
        address donorAddr2 = address(0x11);

        vm.prank(regulator);
        transplant.registerDonorAddress(donorAddr2);

        vm.prank(hospital);
        transplant.registerDonor(donorAddr2, "O", "A3,A11,B7,DR4", "Kidney", "bafyDonor2CID");

        vm.prank(ethicalCommittee);
        transplant.approveDonorEthicalCommittee(2);

        vm.expectRevert(bytes("Primary already matched"));
        vm.prank(llm);
        transplant.createMatch(2, 1, 3, "bafySecondMatchCID");
    }

    function testMatchCreation_FailsIfPrimaryAndBackupAreSame() public {
        _registerDonorAndRecipients12();
        _ethicallyApproveDonorAndRecipients12();

        vm.expectRevert(bytes("Primary and backup must differ"));
        vm.prank(llm);
        transplant.createMatch(1, 1, 1, matchCID);
    }

    function testMatchCreation_FailsIfMatchCIDEmpty() public {
        _registerDonorAndRecipients12();
        _ethicallyApproveDonorAndRecipients12();

        vm.expectRevert(bytes("Empty matchCID"));
        vm.prank(llm);
        transplant.createMatch(1, 1, 2, "");
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Sequential approval tests
    // ─────────────────────────────────────────────────────────────────────────────

    function testSequentialApprovalFlow_Success_Primary() public {
        _registerDonorAndRecipients12();
        _ethicallyApproveDonorAndRecipients12();
        _createMatchPrimaryBackup12();

        vm.prank(medicalTeam);
        transplant.approveMedicalTeam(1);

        vm.prank(hospital);
        transplant.approveHospital(1);

        vm.prank(donorAddr);
        transplant.approveDonor(1);

        vm.prank(recipient1Addr);
        transplant.approveRecipient(1);

        vm.prank(ethicalCommittee);
        transplant.approveFinalTransplant(1);

        assertTrue(transplant.isTransplantApproved(1));
    }

    function testHospitalApproval_FailsIfMedicalNotFirst() public {
        _registerDonorAndRecipients12();
        _ethicallyApproveDonorAndRecipients12();
        _createMatchPrimaryBackup12();

        vm.expectRevert(bytes("Medical approval required first"));
        vm.prank(hospital);
        transplant.approveHospital(1);
    }

    function testDonorApproval_FailsIfHospitalNotFirst() public {
        _registerDonorAndRecipients12();
        _ethicallyApproveDonorAndRecipients12();
        _createMatchPrimaryBackup12();

        vm.prank(medicalTeam);
        transplant.approveMedicalTeam(1);

        vm.expectRevert(bytes("Hospital approval required first"));
        vm.prank(donorAddr);
        transplant.approveDonor(1);
    }

    function testRecipientApproval_FailsIfDonorNotFirst() public {
        _registerDonorAndRecipients12();
        _ethicallyApproveDonorAndRecipients12();
        _createMatchPrimaryBackup12();

        vm.prank(medicalTeam);
        transplant.approveMedicalTeam(1);

        vm.prank(hospital);
        transplant.approveHospital(1);

        vm.expectRevert(bytes("Donor approval required first"));
        vm.prank(recipient1Addr);
        transplant.approveRecipient(1);
    }

    function testEthicsFinalApproval_FailsIfRecipientNotFirst() public {
        _registerDonorAndRecipients12();
        _ethicallyApproveDonorAndRecipients12();
        _createMatchPrimaryBackup12();

        vm.prank(medicalTeam);
        transplant.approveMedicalTeam(1);

        vm.prank(hospital);
        transplant.approveHospital(1);

        vm.prank(donorAddr);
        transplant.approveDonor(1);

        vm.expectRevert(bytes("Recipient approval required first"));
        vm.prank(ethicalCommittee);
        transplant.approveFinalTransplant(1);
    }

    function testDuplicateApprovals_Fail() public {
        _registerDonorAndRecipients12();
        _ethicallyApproveDonorAndRecipients12();
        _createMatchPrimaryBackup12();

        vm.prank(medicalTeam);
        transplant.approveMedicalTeam(1);

        vm.expectRevert(bytes("Medical already approved"));
        vm.prank(medicalTeam);
        transplant.approveMedicalTeam(1);
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Finalization tests
    // ─────────────────────────────────────────────────────────────────────────────

    function testFinalizeMatch_SucceedsOnlyWhenFullyApproved() public {
        _registerDonorAndRecipients12();
        _ethicallyApproveDonorAndRecipients12();
        _createMatchPrimaryBackup12();
        _completeApprovalsPrimary();

        vm.prank(unauthorized);
        transplant.finalizeMatch(1);

        (
            ,
            ,
            ,
            ,
            ,
            ,
            ,
            ,
            ,
            ,
            ,
            ,
            ,
            bool finalized
        ) = transplant.matches(1);

        assertTrue(finalized);
        assertFalse(transplant.donorHasOpenMatch(1));

        (, , , , , , , bool primaryMatched, ) = transplant.recipients(1);
        (, , , , , , , bool backupMatched, ) = transplant.recipients(2);
        assertTrue(primaryMatched);
        assertFalse(backupMatched);
    }

    function testFinalizeMatch_FailsIfNotFullyApproved() public {
        _registerDonorAndRecipients12();
        _ethicallyApproveDonorAndRecipients12();
        _createMatchPrimaryBackup12();

        vm.prank(medicalTeam);
        transplant.approveMedicalTeam(1);

        vm.expectRevert(bytes("Transplant not fully approved"));
        vm.prank(unauthorized);
        transplant.finalizeMatch(1);
    }

    function testApprovalsFailAfterFinalization() public {
        _registerDonorAndRecipients12();
        _ethicallyApproveDonorAndRecipients12();
        _createMatchPrimaryBackup12();
        _completeApprovalsPrimary();

        vm.prank(unauthorized);
        transplant.finalizeMatch(1);

        vm.expectRevert(bytes("Match finalized"));
        vm.prank(hospital);
        transplant.approveHospital(1);
    }

    function testFinalizeMatch_ReleasesOriginalPrimaryAfterBackupPromotion() public {
        _registerDonorAndRecipients12();
        _ethicallyApproveDonorAndRecipients12();
        _createMatchPrimaryBackup12();

        vm.prank(medicalTeam);
        transplant.approveMedicalTeam(1);

        vm.prank(hospital);
        transplant.approveHospital(1);

        vm.prank(donorAddr);
        transplant.approveDonor(1);

        vm.prank(hospital);
        transplant.promoteBackupRecipient(1);

        vm.prank(recipient2Addr);
        transplant.approveRecipient(1);

        vm.prank(ethicalCommittee);
        transplant.approveFinalTransplant(1);

        transplant.finalizeMatch(1);

        (, , , , , , , bool primaryMatched, ) = transplant.recipients(1);
        (, , , , , , , bool backupMatched, ) = transplant.recipients(2);
        assertFalse(primaryMatched);
        assertTrue(backupMatched);
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Backup promotion tests
    // ─────────────────────────────────────────────────────────────────────────────

    function testBackupPromotion_Success_AndRecipientApprovalSwitches() public {
        _registerDonorAndRecipients12();
        _ethicallyApproveDonorAndRecipients12();
        _createMatchPrimaryBackup12();

        vm.prank(medicalTeam);
        transplant.approveMedicalTeam(1);

        vm.prank(hospital);
        transplant.approveHospital(1);

        vm.prank(donorAddr);
        transplant.approveDonor(1);

        vm.prank(hospital);
        transplant.promoteBackupRecipient(1);

        (
            ,
            ,
            ,
            uint256 backupId,
            uint256 activeId,
            bool backupPromoted,
            ,
            ,
            bool medicalApproved,
            bool hospitalApproved,
            bool donorApproved,
            bool activeRecipientApproved,
            bool ethicalCommitteeApproved,
            bool finalized
        ) = transplant.matches(1);

        assertEq(backupId, 2);
        assertEq(activeId, 2);
        assertTrue(backupPromoted);

        assertTrue(medicalApproved);
        assertTrue(hospitalApproved);
        assertTrue(donorApproved);
        assertFalse(activeRecipientApproved);
        assertFalse(ethicalCommitteeApproved);
        assertFalse(finalized);

        vm.expectRevert(bytes("Only active recipient can approve"));
        vm.prank(recipient1Addr);
        transplant.approveRecipient(1);

        vm.prank(recipient2Addr);
        transplant.approveRecipient(1);

        vm.prank(ethicalCommittee);
        transplant.approveFinalTransplant(1);

        assertTrue(transplant.isTransplantApproved(1));
    }

    function testBackupPromotion_FailsIfCalledTwice() public {
        _registerDonorAndRecipients12();
        _ethicallyApproveDonorAndRecipients12();
        _createMatchPrimaryBackup12();

        vm.prank(medicalTeam);
        transplant.promoteBackupRecipient(1);

        vm.expectRevert(bytes("Backup already promoted"));
        vm.prank(hospital);
        transplant.promoteBackupRecipient(1);
    }

    function testBackupPromotion_FailsAfterRecipientApproval() public {
        _registerDonorAndRecipients12();
        _ethicallyApproveDonorAndRecipients12();
        _createMatchPrimaryBackup12();

        vm.prank(medicalTeam);
        transplant.approveMedicalTeam(1);

        vm.prank(hospital);
        transplant.approveHospital(1);

        vm.prank(donorAddr);
        transplant.approveDonor(1);

        vm.prank(recipient1Addr);
        transplant.approveRecipient(1);

        vm.expectRevert(bytes("Recipient already approved"));
        vm.prank(hospital);
        transplant.promoteBackupRecipient(1);
    }

    function testBackupPromotion_FailsAfterFinalEthicsApproval() public {
        _registerDonorAndRecipients12();
        _ethicallyApproveDonorAndRecipients12();
        _createMatchPrimaryBackup12();

        vm.prank(medicalTeam);
        transplant.approveMedicalTeam(1);

        vm.prank(hospital);
        transplant.approveHospital(1);

        vm.prank(donorAddr);
        transplant.approveDonor(1);

        vm.prank(recipient1Addr);
        transplant.approveRecipient(1);

        vm.prank(ethicalCommittee);
        transplant.approveFinalTransplant(1);

        // Recipient approval check is hit first
        vm.expectRevert(bytes("Recipient already approved"));
        vm.prank(medicalTeam);
        transplant.promoteBackupRecipient(1);
    }

    function testPromoteBackupRecipient_FailsForUnauthorizedCaller() public {
        _registerDonorAndRecipients12();
        _ethicallyApproveDonorAndRecipients12();
        _createMatchPrimaryBackup12();

        vm.expectRevert(bytes("Only hospital or medical team"));
        vm.prank(unauthorized);
        transplant.promoteBackupRecipient(1);
    }

    function testChangeRegulator_Success() public {
        address newRegulator = address(0x20);

        vm.prank(regulator);
        transplant.changeRegulator(newRegulator);

        assertEq(transplant.regulator(), newRegulator);
    }

    function testChangeRegulator_FailsForUnauthorizedCaller() public {
        vm.expectRevert(bytes("Only regulator"));
        vm.prank(unauthorized);
        transplant.changeRegulator(address(0x20));
    }
}
