// SPDX-License-Identifier: MIT
pragma solidity 0.8.26;

import "forge-std/Test.sol";
import "../src/TransplantManagement.sol";

contract TransplantManagementV30Test is Test {
    TransplantManagement private transplant;

    address private regulator = address(this);
    address private hospital = address(0x1001);
    address private medicalTeam = address(0x1002);
    address private ethicsCommittee = address(0x1003);
    address private decisionService = address(0x1004);
    address private donorAuthority = address(0x2001);
    address private recipientOne = address(0x3001);
    address private recipientTwo = address(0x3002);
    address private recipientThree = address(0x3003);
    address private outsider = address(0x9999);

    function setUp() public {
        transplant = new TransplantManagement(regulator);
        transplant.setHospital(hospital, true);
        transplant.setMedicalTeam(medicalTeam, true);
        transplant.setEthicsCommittee(ethicsCommittee, true);
        transplant.setDecisionService(decisionService, true);
    }

    function _registerProfiles() internal {
        transplant.registerDonorAuthority(donorAuthority);
        transplant.registerRecipientAddress(recipientOne);
        transplant.registerRecipientAddress(recipientTwo);
        transplant.registerRecipientAddress(recipientThree);

        vm.startPrank(hospital);
        transplant.registerDonor(donorAuthority, "bafy-donor-ciphertext");
        transplant.registerRecipient(recipientOne, "bafy-recipient-1-ciphertext");
        transplant.registerRecipient(recipientTwo, "bafy-recipient-2-ciphertext");
        transplant.registerRecipient(recipientThree, "bafy-recipient-3-ciphertext");
        vm.stopPrank();
    }

    function _makeEligible() internal {
        vm.startPrank(ethicsCommittee);
        transplant.setDonorEligibility(1, true);
        transplant.setRecipientEligibility(1, true);
        transplant.setRecipientEligibility(2, true);
        transplant.setRecipientEligibility(3, true);
        vm.stopPrank();
    }

    function _createMatch() internal {
        vm.prank(decisionService);
        transplant.createMatch(1, 1, 2, "bafy-decision-ciphertext");
    }

    function _createSecondDonorMatchWithReleasedPrimary() internal {
        address secondAuthority = address(0x2002);
        transplant.registerDonorAuthority(secondAuthority);
        vm.prank(hospital);
        transplant.registerDonor(secondAuthority, "bafy-donor-2-ciphertext");
        vm.prank(ethicsCommittee);
        transplant.setDonorEligibility(2, true);
        vm.prank(decisionService);
        transplant.createMatch(2, 1, 3, "bafy-second-decision-ciphertext");
    }

    function _completeApprovals(uint256 matchId, address activeRecipient) internal {
        vm.prank(medicalTeam);
        transplant.approveMedicalTeam(matchId);
        vm.prank(hospital);
        transplant.approveHospital(matchId);
        vm.prank(donorAuthority);
        transplant.approveDonorAuthority(matchId);
        vm.prank(activeRecipient);
        transplant.approveRecipient(matchId);
        vm.prank(ethicsCommittee);
        transplant.approveFinalTransplant(matchId);
    }

    function _recipientState(uint256 recipientId) internal view returns (bool reserved, bool transplanted) {
        (, , , , , reserved, transplanted) = transplant.recipients(recipientId);
    }

    function _donorState(uint256 donorId) internal view returns (bool eligible, bool finalized) {
        (, , , , eligible, finalized) = transplant.donors(donorId);
    }

    function testConstructorRejectsZeroRegulator() public {
        vm.expectRevert(bytes("Invalid regulator"));
        new TransplantManagement(address(0));
    }

    function testRegulatorCanGrantRevokeAndRestoreRole() public {
        address secondHospital = address(0x1010);
        transplant.setHospital(secondHospital, true);
        assertTrue(transplant.registeredHospitals(secondHospital));
        vm.expectRevert(bytes("Account already assigned"));
        transplant.setMedicalTeam(secondHospital, true);
        transplant.setHospital(secondHospital, false);
        assertFalse(transplant.registeredHospitals(secondHospital));
        transplant.setMedicalTeam(secondHospital, true);
        vm.expectRevert(bytes("Account already assigned"));
        transplant.setHospital(secondHospital, true);
        transplant.setMedicalTeam(secondHospital, false);
        transplant.setHospital(secondHospital, true);
        assertTrue(transplant.registeredHospitals(secondHospital));
    }

    function testRoleChangeRejectsUnauthorizedCaller() public {
        vm.expectRevert(bytes("Only regulator"));
        vm.prank(outsider);
        transplant.setDecisionService(outsider, true);
    }

    function testRoleChangeRejectsUnchangedStatus() public {
        vm.expectRevert(bytes("Role status unchanged"));
        transplant.setHospital(hospital, true);
    }

    function testRevokedRoleLosesAccessImmediately() public {
        transplant.setHospital(hospital, false);
        vm.expectRevert(bytes("Only hospital"));
        vm.prank(hospital);
        transplant.registerDonor(donorAuthority, "bafy-ciphertext");
    }

    function testTwoStepRegulatorTransfer() public {
        address nextRegulator = address(0x5001);
        vm.expectRevert(bytes("Account already assigned"));
        transplant.proposeRegulator(hospital);
        transplant.proposeRegulator(nextRegulator);
        assertEq(transplant.pendingRegulator(), nextRegulator);
        vm.prank(nextRegulator);
        transplant.acceptRegulator();
        assertEq(transplant.regulator(), nextRegulator);
        assertEq(transplant.pendingRegulator(), address(0));
    }

    function testOnlyPendingRegulatorCanAccept() public {
        transplant.proposeRegulator(address(0x5001));
        vm.expectRevert(bytes("Only pending regulator"));
        vm.prank(outsider);
        transplant.acceptRegulator();
    }

    function testIdentityAndProfileRegistrationStoresOnlyCIDAndActor() public {
        _registerProfiles();
        (uint256 donorId, address authority, string memory donorCID, bool registered, bool eligible, bool finalized) = transplant.donors(1);
        assertEq(donorId, 1);
        assertEq(authority, donorAuthority);
        assertEq(donorCID, "bafy-donor-ciphertext");
        assertTrue(registered);
        assertFalse(eligible);
        assertFalse(finalized);

        (uint256 recipientId, address account, string memory recipientCID, bool recRegistered, bool recEligible, bool reserved, bool transplanted) = transplant.recipients(1);
        assertEq(recipientId, 1);
        assertEq(account, recipientOne);
        assertEq(recipientCID, "bafy-recipient-1-ciphertext");
        assertTrue(recRegistered);
        assertFalse(recEligible);
        assertFalse(reserved);
        assertFalse(transplanted);
    }

    function testDuplicateIdentityRegistrationFails() public {
        transplant.registerDonorAuthority(donorAuthority);
        vm.expectRevert(bytes("Donor authority already registered"));
        transplant.registerDonorAuthority(donorAuthority);
        vm.expectRevert(bytes("Account already assigned"));
        transplant.registerRecipientAddress(donorAuthority);
        vm.expectRevert(bytes("Account already assigned"));
        transplant.setHospital(donorAuthority, true);
    }

    function testProfileRegistrationRequiresHospital() public {
        transplant.registerDonorAuthority(donorAuthority);
        vm.expectRevert(bytes("Only hospital"));
        vm.prank(outsider);
        transplant.registerDonor(donorAuthority, "bafy-ciphertext");
    }

    function testProfileRegistrationRejectsEmptyCID() public {
        transplant.registerDonorAuthority(donorAuthority);
        vm.expectRevert(bytes("Empty CID"));
        vm.prank(hospital);
        transplant.registerDonor(donorAuthority, "");
    }

    function testProfileUpdateResetsEligibility() public {
        _registerProfiles();
        _makeEligible();
        vm.prank(hospital);
        transplant.updateRecipientProfile(1, "bafy-recipient-1-updated");
        (, , string memory cid, , bool eligible, , ) = transplant.recipients(1);
        assertEq(cid, "bafy-recipient-1-updated");
        assertFalse(eligible);
    }

    function testProfileUpdateFailsDuringOpenMatch() public {
        _registerProfiles();
        _makeEligible();
        _createMatch();
        vm.expectRevert(bytes("Donor has open match"));
        vm.prank(hospital);
        transplant.updateDonorProfile(1, "bafy-new");
        vm.expectRevert(bytes("Recipient reserved"));
        vm.prank(hospital);
        transplant.updateRecipientProfile(1, "bafy-new");
    }

    function testEligibilityCanBeGrantedAndRevoked() public {
        _registerProfiles();
        vm.startPrank(ethicsCommittee);
        transplant.setDonorEligibility(1, true);
        transplant.setDonorEligibility(1, false);
        vm.stopPrank();
        (bool eligible, ) = _donorState(1);
        assertFalse(eligible);
    }

    function testEligibilityRequiresEthicsRole() public {
        _registerProfiles();
        vm.expectRevert(bytes("Only ethics committee"));
        vm.prank(outsider);
        transplant.setRecipientEligibility(1, true);
    }

    function testCreateMatchReservesPrimaryAndBackup() public {
        _registerProfiles();
        _makeEligible();
        _createMatch();
        assertEq(transplant.matchCounter(), 1);
        assertTrue(transplant.donorHasOpenMatch(1));
        (bool primaryReserved, bool primaryTransplanted) = _recipientState(1);
        (bool backupReserved, bool backupTransplanted) = _recipientState(2);
        assertTrue(primaryReserved);
        assertTrue(backupReserved);
        assertFalse(primaryTransplanted);
        assertFalse(backupTransplanted);
    }

    function testCreateMatchRequiresDecisionService() public {
        _registerProfiles();
        _makeEligible();
        vm.expectRevert(bytes("Only decision service"));
        vm.prank(outsider);
        transplant.createMatch(1, 1, 2, "bafy-decision");
    }

    function testCreateMatchRequiresEligibility() public {
        _registerProfiles();
        vm.expectRevert(bytes("Donor not eligible"));
        vm.prank(decisionService);
        transplant.createMatch(1, 1, 2, "bafy-decision");
        vm.prank(ethicsCommittee);
        transplant.setDonorEligibility(1, true);
        vm.expectRevert(bytes("Recipient not eligible"));
        vm.prank(decisionService);
        transplant.createMatch(1, 1, 2, "bafy-decision");
    }

    function testCreateMatchRejectsSameRecipientAndEmptyCID() public {
        _registerProfiles();
        _makeEligible();
        vm.expectRevert(bytes("Recipients must differ"));
        vm.prank(decisionService);
        transplant.createMatch(1, 1, 1, "bafy-decision");
        vm.expectRevert(bytes("Empty CID"));
        vm.prank(decisionService);
        transplant.createMatch(1, 1, 2, "");
    }

    function testDonorCannotHaveTwoOpenMatches() public {
        _registerProfiles();
        _makeEligible();
        _createMatch();
        vm.expectRevert(bytes("Donor has open match"));
        vm.prank(decisionService);
        transplant.createMatch(1, 2, 3, "bafy-second");
    }

    function testReservedRecipientCannotEnterAnotherMatch() public {
        _registerProfiles();
        _makeEligible();
        _createMatch();

        address secondAuthority = address(0x2002);
        transplant.registerDonorAuthority(secondAuthority);
        vm.prank(hospital);
        transplant.registerDonor(secondAuthority, "bafy-donor-2");
        vm.prank(ethicsCommittee);
        transplant.setDonorEligibility(2, true);
        vm.expectRevert(bytes("Recipient reserved"));
        vm.prank(decisionService);
        transplant.createMatch(2, 1, 3, "bafy-second");
    }

    function testSequentialApprovalAndFinalization() public {
        _registerProfiles();
        _makeEligible();
        _createMatch();
        _completeApprovals(1, recipientOne);
        assertTrue(transplant.isTransplantApproved(1));
        vm.prank(hospital);
        transplant.finalizeMatch(1);
        assertFalse(transplant.donorHasOpenMatch(1));
        (, bool donorFinalized) = _donorState(1);
        assertTrue(donorFinalized);
        (bool primaryReserved, bool primaryTransplanted) = _recipientState(1);
        (bool backupReserved, bool backupTransplanted) = _recipientState(2);
        assertFalse(primaryReserved);
        assertTrue(primaryTransplanted);
        assertFalse(backupReserved);
        assertFalse(backupTransplanted);
    }

    function testHospitalApprovalRequiresMedicalFirst() public {
        _registerProfiles();
        _makeEligible();
        _createMatch();
        vm.expectRevert(bytes("Medical approval required first"));
        vm.prank(hospital);
        transplant.approveHospital(1);

        vm.prank(medicalTeam);
        transplant.approveMedicalTeam(1);
        transplant.setMedicalTeam(medicalTeam, false);
        transplant.setHospital(hospital, false);
        transplant.setHospital(medicalTeam, true);
        vm.expectRevert(bytes("Approver already used"));
        vm.prank(medicalTeam);
        transplant.approveHospital(1);
    }

    function testDonorAuthorityApprovalRequiresHospitalFirst() public {
        _registerProfiles();
        _makeEligible();
        _createMatch();
        vm.expectRevert(bytes("Hospital approval required first"));
        vm.prank(donorAuthority);
        transplant.approveDonorAuthority(1);
    }

    function testOnlyBoundDonorAuthorityCanApprove() public {
        _registerProfiles();
        _makeEligible();
        _createMatch();
        vm.prank(medicalTeam);
        transplant.approveMedicalTeam(1);
        vm.prank(hospital);
        transplant.approveHospital(1);
        vm.expectRevert(bytes("Only donor authority"));
        vm.prank(outsider);
        transplant.approveDonorAuthority(1);
    }

    function testRecipientApprovalRequiresDonorAuthorityFirst() public {
        _registerProfiles();
        _makeEligible();
        _createMatch();
        vm.expectRevert(bytes("Donor authority approval required first"));
        vm.prank(recipientOne);
        transplant.approveRecipient(1);
    }

    function testOnlyActiveRecipientCanApprove() public {
        _registerProfiles();
        _makeEligible();
        _createMatch();
        vm.prank(medicalTeam);
        transplant.approveMedicalTeam(1);
        vm.prank(hospital);
        transplant.approveHospital(1);
        vm.prank(donorAuthority);
        transplant.approveDonorAuthority(1);
        vm.expectRevert(bytes("Only active recipient"));
        vm.prank(recipientTwo);
        transplant.approveRecipient(1);
    }

    function testFinalEthicsApprovalRequiresRecipientFirst() public {
        _registerProfiles();
        _makeEligible();
        _createMatch();
        vm.expectRevert(bytes("Recipient approval required first"));
        vm.prank(ethicsCommittee);
        transplant.approveFinalTransplant(1);
    }

    function testDuplicateApprovalFails() public {
        _registerProfiles();
        _makeEligible();
        _createMatch();
        vm.prank(medicalTeam);
        transplant.approveMedicalTeam(1);
        vm.expectRevert(bytes("Medical approval already recorded"));
        vm.prank(medicalTeam);
        transplant.approveMedicalTeam(1);
    }

    function testPromotionReleasesPrimaryAndResetsAllApprovals() public {
        _registerProfiles();
        _makeEligible();
        _createMatch();
        vm.prank(medicalTeam);
        transplant.approveMedicalTeam(1);
        vm.prank(hospital);
        transplant.approveHospital(1);
        vm.prank(donorAuthority);
        transplant.approveDonorAuthority(1);
        vm.prank(hospital);
        transplant.promoteBackupRecipient(1, "bafy-promotion-ciphertext");

        (bool oldPrimaryReserved, ) = _recipientState(1);
        (bool backupReserved, ) = _recipientState(2);
        assertFalse(oldPrimaryReserved);
        assertTrue(backupReserved);
        (
            , , , , uint256 activeRecipientId, bool promoted, , , ,
            bool medicalApproved, bool hospitalApproved, bool donorApproved,
            bool recipientApproved, bool ethicsApproved, ,
        ) = transplant.matches(1);
        assertEq(activeRecipientId, 2);
        assertTrue(promoted);
        assertEq(transplant.promotionCIDs(1), "bafy-promotion-ciphertext");
        assertEq(transplant.approvalRounds(1), 2);
        assertFalse(medicalApproved);
        assertFalse(hospitalApproved);
        assertFalse(donorApproved);
        assertFalse(recipientApproved);
        assertFalse(ethicsApproved);
    }

    function testPromotionCanOnlyHappenOnce() public {
        _registerProfiles();
        _makeEligible();
        _createMatch();
        vm.prank(hospital);
        transplant.promoteBackupRecipient(1, "bafy-promotion-ciphertext");
        vm.expectRevert(bytes("Backup already promoted"));
        vm.prank(medicalTeam);
        transplant.promoteBackupRecipient(1, "bafy-second-promotion");
    }

    function testPromotionRequiresEncryptedReasonCID() public {
        _registerProfiles();
        _makeEligible();
        _createMatch();
        vm.expectRevert(bytes("Empty CID"));
        vm.prank(hospital);
        transplant.promoteBackupRecipient(1, "");
    }

    function testUnauthorizedPromotionFails() public {
        _registerProfiles();
        _makeEligible();
        _createMatch();
        vm.expectRevert(bytes("Only hospital or medical team"));
        vm.prank(outsider);
        transplant.promoteBackupRecipient(1, "bafy-promotion-ciphertext");
    }

    function testPromotedBackupMustCompleteFreshApprovalSequence() public {
        _registerProfiles();
        _makeEligible();
        _createMatch();
        vm.prank(hospital);
        transplant.promoteBackupRecipient(1, "bafy-promotion-ciphertext");
        _completeApprovals(1, recipientTwo);
        vm.prank(medicalTeam);
        transplant.finalizeMatch(1);
        (, bool originalTransplanted) = _recipientState(1);
        (, bool backupTransplanted) = _recipientState(2);
        assertFalse(originalTransplanted);
        assertTrue(backupTransplanted);
    }

    function testCancellingPromotedMatchDoesNotReleaseReservationOwnedByAnotherMatch() public {
        _registerProfiles();
        _makeEligible();
        _createMatch();
        vm.prank(hospital);
        transplant.promoteBackupRecipient(1, "bafy-promotion-ciphertext");
        _createSecondDonorMatchWithReleasedPrimary();

        vm.prank(medicalTeam);
        transplant.cancelMatch(1, "bafy-cancel-promoted-match");

        (bool reusedPrimaryReserved, ) = _recipientState(1);
        (bool secondBackupReserved, ) = _recipientState(3);
        assertTrue(reusedPrimaryReserved);
        assertTrue(secondBackupReserved);
    }

    function testFinalizingPromotedMatchDoesNotReleaseReservationOwnedByAnotherMatch() public {
        _registerProfiles();
        _makeEligible();
        _createMatch();
        vm.prank(hospital);
        transplant.promoteBackupRecipient(1, "bafy-promotion-ciphertext");
        _createSecondDonorMatchWithReleasedPrimary();
        _completeApprovals(1, recipientTwo);

        vm.prank(medicalTeam);
        transplant.finalizeMatch(1);

        (bool reusedPrimaryReserved, ) = _recipientState(1);
        (bool secondBackupReserved, ) = _recipientState(3);
        assertTrue(reusedPrimaryReserved);
        assertTrue(secondBackupReserved);
    }

    function testCancellationReleasesReservationsAndAllowsNewMatch() public {
        _registerProfiles();
        _makeEligible();
        _createMatch();
        vm.prank(medicalTeam);
        transplant.cancelMatch(1, "bafy-cancellation-ciphertext");
        assertFalse(transplant.donorHasOpenMatch(1));
        (bool primaryReserved, ) = _recipientState(1);
        (bool backupReserved, ) = _recipientState(2);
        assertFalse(primaryReserved);
        assertFalse(backupReserved);
        vm.prank(decisionService);
        transplant.createMatch(1, 2, 3, "bafy-new-decision");
        assertEq(transplant.matchCounter(), 2);
    }

    function testCancellationRequiresNonemptyCIDAndAuthorizedRole() public {
        _registerProfiles();
        _makeEligible();
        _createMatch();
        vm.expectRevert(bytes("Empty CID"));
        vm.prank(hospital);
        transplant.cancelMatch(1, "");
        vm.expectRevert(bytes("Only hospital or medical team"));
        vm.prank(outsider);
        transplant.cancelMatch(1, "bafy-cancel");
    }

    function testNoActionAfterCancellation() public {
        _registerProfiles();
        _makeEligible();
        _createMatch();
        vm.prank(hospital);
        transplant.cancelMatch(1, "bafy-cancel");
        vm.expectRevert(bytes("Match cancelled"));
        vm.prank(medicalTeam);
        transplant.approveMedicalTeam(1);
    }

    function testFinalizeRequiresFullApprovalAndAuthorizedRole() public {
        _registerProfiles();
        _makeEligible();
        _createMatch();
        vm.expectRevert(bytes("Match is not fully approved"));
        vm.prank(hospital);
        transplant.finalizeMatch(1);
        _completeApprovals(1, recipientOne);
        vm.expectRevert(bytes("Only hospital or medical team"));
        vm.prank(outsider);
        transplant.finalizeMatch(1);
    }

    function testNoActionAfterFinalization() public {
        _registerProfiles();
        _makeEligible();
        _createMatch();
        _completeApprovals(1, recipientOne);
        vm.prank(hospital);
        transplant.finalizeMatch(1);
        vm.expectRevert(bytes("Match finalized"));
        vm.prank(medicalTeam);
        transplant.approveMedicalTeam(1);
        vm.expectRevert(bytes("Donor finalized"));
        vm.prank(ethicsCommittee);
        transplant.setDonorEligibility(1, false);
    }
}
