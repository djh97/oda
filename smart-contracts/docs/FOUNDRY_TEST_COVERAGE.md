# Foundry Smart-Contract Test Coverage

Last verified: 2026-09-09

## Scope

This document explains all 41 Foundry test functions reported in the manuscript.
The authoritative executable specification is
`test/TransplantManagementV30.t.sol`, which tests
`src/TransplantManagement.sol` with Solidity 0.8.26. The retained run reports
41 passed, 0 failed, and 0 skipped tests.

Each test starts from a fresh contract deployment. The shared setup assigns
separate synthetic addresses to the regulator, hospital, medical team, ethics
committee, decision service, donor authority, recipients, and an unauthorized
account. Helper functions register encrypted-content identifiers, set workflow
eligibility, create a match, and execute the ordered approval path. No patient
records or human decisions are used.

One Foundry test may exercise several related assertions or expected reverts.
Accordingly, the 41 functions below cover more than 41 individual conditions.

## Governance And Roles

| # | Foundry test | Verified behavior |
|---:|---|---|
| 1 | `testConstructorRejectsZeroRegulator` | Deployment reverts when the initial regulator is the zero address. |
| 2 | `testRegulatorCanGrantRevokeAndRestoreRole` | The regulator can grant, revoke, reassign, and restore roles, while one address cannot hold conflicting active roles. |
| 3 | `testRoleChangeRejectsUnauthorizedCaller` | An unauthorized account cannot grant or revoke a system role. |
| 4 | `testRoleChangeRejectsUnchangedStatus` | Repeating the current role status is rejected instead of emitting a redundant state change. |
| 5 | `testRevokedRoleLosesAccessImmediately` | A revoked hospital account immediately loses access to hospital-only registration. |
| 6 | `testTwoStepRegulatorTransfer` | Regulator transfer requires proposal and acceptance, rejects an already assigned account, updates the regulator, and clears the pending address. |
| 7 | `testOnlyPendingRegulatorCanAccept` | An account other than the nominated successor cannot accept the regulator role. |

## Identity And Profiles

| # | Foundry test | Verified behavior |
|---:|---|---|
| 8 | `testIdentityAndProfileRegistrationStoresOnlyCIDAndActor` | Donor and recipient registration stores the assigned ID, bound actor address, encrypted-content CID, and lifecycle flags without storing profile attributes. |
| 9 | `testDuplicateIdentityRegistrationFails` | Duplicate donor-authority registration and conflicting identity or role assignment are rejected. |
| 10 | `testProfileRegistrationRequiresHospital` | Only an authorized hospital can register a donor profile. |
| 11 | `testProfileRegistrationRejectsEmptyCID` | Profile registration rejects an empty content identifier. |
| 12 | `testProfileUpdateResetsEligibility` | Updating a recipient profile replaces its CID and clears prior eligibility. |
| 13 | `testProfileUpdateFailsDuringOpenMatch` | Donor and recipient profiles cannot be changed while they participate in an open match. |

## Workflow Eligibility

| # | Foundry test | Verified behavior |
|---:|---|---|
| 14 | `testEligibilityCanBeGrantedAndRevoked` | The ethics-role account can grant and later revoke donor workflow eligibility. |
| 15 | `testEligibilityRequiresEthicsRole` | An unauthorized account cannot change recipient workflow eligibility. |

## Match Creation And Reservations

| # | Foundry test | Verified behavior |
|---:|---|---|
| 16 | `testCreateMatchReservesPrimaryAndBackup` | Match creation increments the match counter, marks the donor as having an open match, and reserves both selected recipients. |
| 17 | `testCreateMatchRequiresDecisionService` | Only the decision-service role can submit a match. |
| 18 | `testCreateMatchRequiresEligibility` | Match creation fails when the donor or either selected recipient lacks workflow eligibility. |
| 19 | `testCreateMatchRejectsSameRecipientAndEmptyCID` | The primary and backup must differ, and the encrypted decision CID must be nonempty. |
| 20 | `testDonorCannotHaveTwoOpenMatches` | A donor cannot be assigned to a second match while its first match remains open. |
| 21 | `testReservedRecipientCannotEnterAnotherMatch` | A recipient reserved for one donor cannot be reused in another donor's open match. |

## Ordered Approvals And Finalization

| # | Foundry test | Verified behavior |
|---:|---|---|
| 22 | `testSequentialApprovalAndFinalization` | The complete approval sequence enables finalization, closes the donor, marks the active recipient transplanted, and releases both reservations. |
| 23 | `testHospitalApprovalRequiresMedicalFirst` | Hospital approval requires prior medical-team approval, and one address cannot satisfy two approval stages in the same round. |
| 24 | `testDonorAuthorityApprovalRequiresHospitalFirst` | Donor-authority approval is rejected until hospital approval is recorded. |
| 25 | `testOnlyBoundDonorAuthorityCanApprove` | Only the authority address bound to the matched donor can provide donor approval. |
| 26 | `testRecipientApprovalRequiresDonorAuthorityFirst` | Recipient approval is rejected until donor-authority approval is recorded. |
| 27 | `testOnlyActiveRecipientCanApprove` | The backup or another account cannot approve in place of the active recipient. |
| 28 | `testFinalEthicsApprovalRequiresRecipientFirst` | Final ethics approval is rejected until recipient approval is recorded. |
| 29 | `testDuplicateApprovalFails` | An approval stage cannot be recorded twice in the same approval round. |

## Backup Promotion And Reservation Ownership

| # | Foundry test | Verified behavior |
|---:|---|---|
| 30 | `testPromotionReleasesPrimaryAndResetsAllApprovals` | Backup promotion releases the former primary, keeps the backup reserved, stores the encrypted reason CID, starts a new approval round, and clears all prior approvals. |
| 31 | `testPromotionCanOnlyHappenOnce` | A match cannot promote its backup more than once. |
| 32 | `testPromotionRequiresEncryptedReasonCID` | Backup promotion requires a nonempty encrypted reason CID. |
| 33 | `testUnauthorizedPromotionFails` | Only the hospital or medical team can promote the backup. |
| 34 | `testPromotedBackupMustCompleteFreshApprovalSequence` | The promoted backup completes a fresh approval sequence before finalization and is recorded as the transplanted recipient. |
| 35 | `testCancellingPromotedMatchDoesNotReleaseReservationOwnedByAnotherMatch` | Cancelling a promoted match does not clear a former primary's reservation after that recipient has been reserved by another match. |
| 36 | `testFinalizingPromotedMatchDoesNotReleaseReservationOwnedByAnotherMatch` | Finalizing a promoted match does not clear a former primary's reservation after that recipient has been reserved by another match. |

## Cancellation And Terminal States

| # | Foundry test | Verified behavior |
|---:|---|---|
| 37 | `testCancellationReleasesReservationsAndAllowsNewMatch` | Cancellation closes the donor's open match, releases both reservations, and permits a subsequent match. |
| 38 | `testCancellationRequiresNonemptyCIDAndAuthorizedRole` | Cancellation requires a nonempty reason CID and an authorized hospital or medical-team caller. |
| 39 | `testNoActionAfterCancellation` | Approval calls are rejected after cancellation. |
| 40 | `testFinalizeRequiresFullApprovalAndAuthorizedRole` | Finalization requires the full approval sequence and an authorized hospital or medical-team caller. |
| 41 | `testNoActionAfterFinalization` | Further approvals and donor-eligibility changes are rejected after finalization. |

## Reproduction And Evidence

Run the current suite from the `smart-contracts` directory with Foundry and
Solidity 0.8.26.

```powershell
forge test --match-contract TransplantManagementV30Test -vv
```

The complete retained console output is
`test-output/foundry_v30_full.txt`. The corresponding gas-report run is
`test-output/foundry_v30_gas_report.txt`. The manuscript figure is a native
terminal capture of the 41-test run, while these text files and the Solidity
source remain authoritative.

## Interpretation Limits

The suite covers the principal authorization, reservation, approval-order,
promotion, cancellation, and terminal-state paths implemented by the contract.
It does not constitute a manual audit or formal proof, and it does not exhaust
all transaction sequences or adversarial inputs. Slither static analysis and
the Python application tests provide separate evidence and do not expand the
claims made for these 41 contract tests.
