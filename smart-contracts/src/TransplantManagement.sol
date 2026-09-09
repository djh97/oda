// SPDX-License-Identifier: MIT
pragma solidity 0.8.26;

/// @title TransplantManagement
/// @notice Prototype governance state machine for synthetic transplant cases.
/// @dev Clinical attributes and rationales are encrypted off-chain. Public CIDs
///      refer only to authenticated ciphertext.
contract TransplantManagement {
    bytes32 public constant HOSPITAL_ROLE = keccak256("HOSPITAL_ROLE");
    bytes32 public constant MEDICAL_TEAM_ROLE = keccak256("MEDICAL_TEAM_ROLE");
    bytes32 public constant ETHICS_COMMITTEE_ROLE = keccak256("ETHICS_COMMITTEE_ROLE");
    bytes32 public constant DECISION_SERVICE_ROLE = keccak256("DECISION_SERVICE_ROLE");

    address public regulator;
    address public pendingRegulator;

    mapping(address => bool) public registeredHospitals;
    mapping(address => bool) public registeredMedicalTeams;
    mapping(address => bool) public registeredEthicsCommittee;
    mapping(address => bool) public authorizedDecisionServices;

    struct Donor {
        uint256 donorId;
        address donorAuthority;
        string profileCID;
        bool registered;
        bool ethicallyEligible;
        bool finalized;
    }

    struct Recipient {
        uint256 recipientId;
        address recipientAddress;
        string profileCID;
        bool registered;
        bool ethicallyEligible;
        bool reserved;
        bool transplanted;
    }

    struct Match {
        uint256 matchId;
        uint256 donorId;
        uint256 primaryRecipientId;
        uint256 backupRecipientId;
        uint256 activeRecipientId;
        bool backupPromoted;
        address recordedBy;
        string decisionCID;
        string cancellationCID;
        bool medicalApproved;
        bool hospitalApproved;
        bool donorAuthorityApproved;
        bool activeRecipientApproved;
        bool ethicsCommitteeApproved;
        bool finalized;
        bool cancelled;
    }

    mapping(uint256 => Donor) public donors;
    mapping(uint256 => Recipient) public recipients;
    mapping(uint256 => Match) public matches;
    mapping(address => uint256) public registeredDonorAuthorities;
    mapping(address => uint256) public registeredRecipientAddresses;
    mapping(uint256 => bool) public donorHasOpenMatch;
    mapping(uint256 => string) public promotionCIDs;
    mapping(uint256 => uint256) public approvalRounds;
    mapping(uint256 => mapping(address => uint256)) private approvalActorRounds;

    uint256 public donorCounter;
    uint256 public recipientCounter;
    uint256 public matchCounter;

    event RegulatorProposed(address indexed currentRegulator, address indexed proposedRegulator);
    event RegulatorChanged(address indexed oldRegulator, address indexed newRegulator);
    event RoleStatusChanged(bytes32 indexed role, address indexed account, bool enabled);
    event DonorAuthorityRegistered(address indexed donorAuthority, uint256 indexed donorId);
    event RecipientAddressRegistered(address indexed recipientAddress, uint256 indexed recipientId);
    event DonorProfileRegistered(uint256 indexed donorId, address indexed donorAuthority, string profileCID);
    event RecipientProfileRegistered(uint256 indexed recipientId, address indexed recipientAddress, string profileCID);
    event ProfileCIDUpdated(uint256 indexed id, bool indexed isDonor, string profileCID);
    event EligibilityChanged(uint256 indexed id, bool indexed isDonor, bool eligible);
    event MatchCreated(
        uint256 indexed matchId,
        uint256 indexed donorId,
        uint256 indexed primaryRecipientId,
        uint256 backupRecipientId,
        string decisionCID,
        address recordedBy
    );
    event BackupRecipientPromoted(
        uint256 indexed matchId,
        uint256 indexed oldActiveRecipientId,
        uint256 indexed newActiveRecipientId,
        string promotionCID,
        address promotedBy
    );
    event ApprovalGranted(uint256 indexed matchId, bytes32 indexed approvalRole, address indexed approver);
    event MatchCancelled(uint256 indexed matchId, string cancellationCID, address indexed cancelledBy);
    event MatchFinalized(uint256 indexed matchId, uint256 indexed activeRecipientId, address indexed finalizedBy);

    modifier onlyRegulator() {
        require(msg.sender == regulator, "Only regulator");
        _;
    }

    modifier onlyHospital() {
        require(registeredHospitals[msg.sender], "Only hospital");
        _;
    }

    modifier onlyMedicalTeam() {
        require(registeredMedicalTeams[msg.sender], "Only medical team");
        _;
    }

    modifier onlyEthicsCommittee() {
        require(registeredEthicsCommittee[msg.sender], "Only ethics committee");
        _;
    }

    modifier onlyDecisionService() {
        require(authorizedDecisionServices[msg.sender], "Only decision service");
        _;
    }

    modifier onlyHospitalOrMedicalTeam() {
        require(
            registeredHospitals[msg.sender] || registeredMedicalTeams[msg.sender],
            "Only hospital or medical team"
        );
        _;
    }

    constructor(address initialRegulator) {
        require(initialRegulator != address(0), "Invalid regulator");
        regulator = initialRegulator;
    }

    function _requireNonemptyCID(string calldata cid) internal pure {
        require(bytes(cid).length > 0, "Empty CID");
    }

    function _requireMatchOpen(uint256 matchId) internal view {
        Match storage current = matches[matchId];
        require(current.matchId != 0, "Match does not exist");
        require(!current.finalized, "Match finalized");
        require(!current.cancelled, "Match cancelled");
    }

    function _resetApprovals(Match storage current) internal {
        current.medicalApproved = false;
        current.hospitalApproved = false;
        current.donorAuthorityApproved = false;
        current.activeRecipientApproved = false;
        current.ethicsCommitteeApproved = false;
    }

    function _recordApprovalActor(uint256 matchId) internal {
        uint256 currentRound = approvalRounds[matchId];
        require(currentRound != 0, "Approval round missing");
        require(
            approvalActorRounds[matchId][msg.sender] != currentRound,
            "Approver already used"
        );
        approvalActorRounds[matchId][msg.sender] = currentRound;
    }

    function _requireUnassignedAccount(address account) internal view {
        require(
            account != regulator &&
                account != pendingRegulator &&
                !registeredHospitals[account] &&
                !registeredMedicalTeams[account] &&
                !registeredEthicsCommittee[account] &&
                !authorizedDecisionServices[account] &&
                registeredDonorAuthorities[account] == 0 &&
                registeredRecipientAddresses[account] == 0,
            "Account already assigned"
        );
    }

    function _releaseReservations(Match storage current) internal {
        if (!current.backupPromoted) {
            recipients[current.primaryRecipientId].reserved = false;
        }
        recipients[current.backupRecipientId].reserved = false;
    }

    function proposeRegulator(address proposedRegulator) external onlyRegulator {
        require(proposedRegulator != address(0), "Invalid regulator");
        require(proposedRegulator != regulator, "Already regulator");
        _requireUnassignedAccount(proposedRegulator);
        pendingRegulator = proposedRegulator;
        emit RegulatorProposed(regulator, proposedRegulator);
    }

    function acceptRegulator() external {
        require(msg.sender == pendingRegulator, "Only pending regulator");
        address oldRegulator = regulator;
        regulator = pendingRegulator;
        pendingRegulator = address(0);
        emit RegulatorChanged(oldRegulator, regulator);
    }

    function setHospital(address account, bool enabled) external onlyRegulator {
        _setRole(registeredHospitals, HOSPITAL_ROLE, account, enabled);
    }

    function setMedicalTeam(address account, bool enabled) external onlyRegulator {
        _setRole(registeredMedicalTeams, MEDICAL_TEAM_ROLE, account, enabled);
    }

    function setEthicsCommittee(address account, bool enabled) external onlyRegulator {
        _setRole(registeredEthicsCommittee, ETHICS_COMMITTEE_ROLE, account, enabled);
    }

    function setDecisionService(address account, bool enabled) external onlyRegulator {
        _setRole(authorizedDecisionServices, DECISION_SERVICE_ROLE, account, enabled);
    }

    function _setRole(
        mapping(address => bool) storage roleMembers,
        bytes32 role,
        address account,
        bool enabled
    ) internal {
        require(account != address(0), "Invalid role account");
        require(roleMembers[account] != enabled, "Role status unchanged");
        if (enabled) {
            _requireUnassignedAccount(account);
        }
        roleMembers[account] = enabled;
        emit RoleStatusChanged(role, account, enabled);
    }

    function registerDonorAuthority(address donorAuthority) external onlyRegulator returns (uint256 donorId) {
        require(donorAuthority != address(0), "Invalid donor authority");
        require(registeredDonorAuthorities[donorAuthority] == 0, "Donor authority already registered");
        _requireUnassignedAccount(donorAuthority);
        donorId = ++donorCounter;
        registeredDonorAuthorities[donorAuthority] = donorId;
        emit DonorAuthorityRegistered(donorAuthority, donorId);
    }

    function registerRecipientAddress(address recipientAddress) external onlyRegulator returns (uint256 recipientId) {
        require(recipientAddress != address(0), "Invalid recipient address");
        require(registeredRecipientAddresses[recipientAddress] == 0, "Recipient already registered");
        _requireUnassignedAccount(recipientAddress);
        recipientId = ++recipientCounter;
        registeredRecipientAddresses[recipientAddress] = recipientId;
        emit RecipientAddressRegistered(recipientAddress, recipientId);
    }

    function registerDonor(address donorAuthority, string calldata profileCID) external onlyHospital {
        uint256 donorId = registeredDonorAuthorities[donorAuthority];
        require(donorId != 0, "Donor authority not registered");
        require(!donors[donorId].registered, "Donor profile already registered");
        _requireNonemptyCID(profileCID);
        donors[donorId] = Donor({
            donorId: donorId,
            donorAuthority: donorAuthority,
            profileCID: profileCID,
            registered: true,
            ethicallyEligible: false,
            finalized: false
        });
        emit DonorProfileRegistered(donorId, donorAuthority, profileCID);
    }

    function registerRecipient(address recipientAddress, string calldata profileCID) external onlyHospital {
        uint256 recipientId = registeredRecipientAddresses[recipientAddress];
        require(recipientId != 0, "Recipient address not registered");
        require(!recipients[recipientId].registered, "Recipient profile already registered");
        _requireNonemptyCID(profileCID);
        recipients[recipientId] = Recipient({
            recipientId: recipientId,
            recipientAddress: recipientAddress,
            profileCID: profileCID,
            registered: true,
            ethicallyEligible: false,
            reserved: false,
            transplanted: false
        });
        emit RecipientProfileRegistered(recipientId, recipientAddress, profileCID);
    }

    function updateDonorProfile(uint256 donorId, string calldata profileCID) external onlyHospital {
        Donor storage donor = donors[donorId];
        require(donor.registered, "Donor not registered");
        require(!donor.finalized, "Donor finalized");
        require(!donorHasOpenMatch[donorId], "Donor has open match");
        _requireNonemptyCID(profileCID);
        donor.profileCID = profileCID;
        donor.ethicallyEligible = false;
        emit ProfileCIDUpdated(donorId, true, profileCID);
        emit EligibilityChanged(donorId, true, false);
    }

    function updateRecipientProfile(uint256 recipientId, string calldata profileCID) external onlyHospital {
        Recipient storage recipient = recipients[recipientId];
        require(recipient.registered, "Recipient not registered");
        require(!recipient.reserved, "Recipient reserved");
        require(!recipient.transplanted, "Recipient transplanted");
        _requireNonemptyCID(profileCID);
        recipient.profileCID = profileCID;
        recipient.ethicallyEligible = false;
        emit ProfileCIDUpdated(recipientId, false, profileCID);
        emit EligibilityChanged(recipientId, false, false);
    }

    function setDonorEligibility(uint256 donorId, bool eligible) external onlyEthicsCommittee {
        Donor storage donor = donors[donorId];
        require(donor.registered, "Donor not registered");
        require(!donor.finalized, "Donor finalized");
        require(!donorHasOpenMatch[donorId], "Donor has open match");
        require(donor.ethicallyEligible != eligible, "Eligibility unchanged");
        donor.ethicallyEligible = eligible;
        emit EligibilityChanged(donorId, true, eligible);
    }

    function setRecipientEligibility(uint256 recipientId, bool eligible) external onlyEthicsCommittee {
        Recipient storage recipient = recipients[recipientId];
        require(recipient.registered, "Recipient not registered");
        require(!recipient.reserved, "Recipient reserved");
        require(!recipient.transplanted, "Recipient transplanted");
        require(recipient.ethicallyEligible != eligible, "Eligibility unchanged");
        recipient.ethicallyEligible = eligible;
        emit EligibilityChanged(recipientId, false, eligible);
    }

    function createMatch(
        uint256 donorId,
        uint256 primaryRecipientId,
        uint256 backupRecipientId,
        string calldata decisionCID
    ) external onlyDecisionService returns (uint256 matchId) {
        Donor storage donor = donors[donorId];
        Recipient storage primary = recipients[primaryRecipientId];
        Recipient storage backup = recipients[backupRecipientId];
        require(donor.registered, "Donor not registered");
        require(donor.ethicallyEligible, "Donor not eligible");
        require(!donor.finalized, "Donor finalized");
        require(!donorHasOpenMatch[donorId], "Donor has open match");
        require(primaryRecipientId != 0 && backupRecipientId != 0, "Invalid recipient IDs");
        require(primaryRecipientId != backupRecipientId, "Recipients must differ");
        require(primary.registered && backup.registered, "Recipient not registered");
        require(primary.ethicallyEligible && backup.ethicallyEligible, "Recipient not eligible");
        require(!primary.reserved && !backup.reserved, "Recipient reserved");
        require(!primary.transplanted && !backup.transplanted, "Recipient transplanted");
        _requireNonemptyCID(decisionCID);

        matchId = ++matchCounter;
        Match storage current = matches[matchId];
        current.matchId = matchId;
        current.donorId = donorId;
        current.primaryRecipientId = primaryRecipientId;
        current.backupRecipientId = backupRecipientId;
        current.activeRecipientId = primaryRecipientId;
        current.recordedBy = msg.sender;
        current.decisionCID = decisionCID;
        approvalRounds[matchId] = 1;
        donorHasOpenMatch[donorId] = true;
        primary.reserved = true;
        backup.reserved = true;
        emit MatchCreated(
            matchId,
            donorId,
            primaryRecipientId,
            backupRecipientId,
            decisionCID,
            msg.sender
        );
    }

    function approveMedicalTeam(uint256 matchId) external onlyMedicalTeam {
        _requireMatchOpen(matchId);
        Match storage current = matches[matchId];
        require(!current.medicalApproved, "Medical approval already recorded");
        _recordApprovalActor(matchId);
        current.medicalApproved = true;
        emit ApprovalGranted(matchId, MEDICAL_TEAM_ROLE, msg.sender);
    }

    function approveHospital(uint256 matchId) external onlyHospital {
        _requireMatchOpen(matchId);
        Match storage current = matches[matchId];
        require(current.medicalApproved, "Medical approval required first");
        require(!current.hospitalApproved, "Hospital approval already recorded");
        _recordApprovalActor(matchId);
        current.hospitalApproved = true;
        emit ApprovalGranted(matchId, HOSPITAL_ROLE, msg.sender);
    }

    function approveDonorAuthority(uint256 matchId) external {
        _requireMatchOpen(matchId);
        Match storage current = matches[matchId];
        require(current.hospitalApproved, "Hospital approval required first");
        require(!current.donorAuthorityApproved, "Donor authority approval already recorded");
        require(msg.sender == donors[current.donorId].donorAuthority, "Only donor authority");
        _recordApprovalActor(matchId);
        current.donorAuthorityApproved = true;
        emit ApprovalGranted(matchId, keccak256("DONOR_AUTHORITY"), msg.sender);
    }

    function approveRecipient(uint256 matchId) external {
        _requireMatchOpen(matchId);
        Match storage current = matches[matchId];
        require(current.donorAuthorityApproved, "Donor authority approval required first");
        require(!current.activeRecipientApproved, "Recipient approval already recorded");
        require(msg.sender == recipients[current.activeRecipientId].recipientAddress, "Only active recipient");
        _recordApprovalActor(matchId);
        current.activeRecipientApproved = true;
        emit ApprovalGranted(matchId, keccak256("RECIPIENT"), msg.sender);
    }

    function approveFinalTransplant(uint256 matchId) external onlyEthicsCommittee {
        _requireMatchOpen(matchId);
        Match storage current = matches[matchId];
        require(current.activeRecipientApproved, "Recipient approval required first");
        require(!current.ethicsCommitteeApproved, "Ethics approval already recorded");
        _recordApprovalActor(matchId);
        current.ethicsCommitteeApproved = true;
        emit ApprovalGranted(matchId, ETHICS_COMMITTEE_ROLE, msg.sender);
    }

    function promoteBackupRecipient(
        uint256 matchId,
        string calldata promotionCID
    ) external onlyHospitalOrMedicalTeam {
        _requireMatchOpen(matchId);
        _requireNonemptyCID(promotionCID);
        Match storage current = matches[matchId];
        require(!current.backupPromoted, "Backup already promoted");
        require(!current.ethicsCommitteeApproved, "Final ethics approval already recorded");
        uint256 oldActive = current.activeRecipientId;
        recipients[oldActive].reserved = false;
        current.activeRecipientId = current.backupRecipientId;
        current.backupPromoted = true;
        promotionCIDs[matchId] = promotionCID;
        _resetApprovals(current);
        approvalRounds[matchId] += 1;
        emit BackupRecipientPromoted(
            matchId,
            oldActive,
            current.activeRecipientId,
            promotionCID,
            msg.sender
        );
    }

    function cancelMatch(uint256 matchId, string calldata cancellationCID) external onlyHospitalOrMedicalTeam {
        _requireMatchOpen(matchId);
        _requireNonemptyCID(cancellationCID);
        Match storage current = matches[matchId];
        current.cancelled = true;
        current.cancellationCID = cancellationCID;
        donorHasOpenMatch[current.donorId] = false;
        _releaseReservations(current);
        emit MatchCancelled(matchId, cancellationCID, msg.sender);
    }

    function isTransplantApproved(uint256 matchId) public view returns (bool) {
        Match storage current = matches[matchId];
        return (
            current.matchId != 0 &&
            !current.cancelled &&
            !current.finalized &&
            current.medicalApproved &&
            current.hospitalApproved &&
            current.donorAuthorityApproved &&
            current.activeRecipientApproved &&
            current.ethicsCommitteeApproved
        );
    }

    function finalizeMatch(uint256 matchId) external onlyHospitalOrMedicalTeam {
        _requireMatchOpen(matchId);
        require(isTransplantApproved(matchId), "Match is not fully approved");
        Match storage current = matches[matchId];
        current.finalized = true;
        donorHasOpenMatch[current.donorId] = false;
        donors[current.donorId].finalized = true;
        _releaseReservations(current);
        recipients[current.activeRecipientId].transplanted = true;
        emit MatchFinalized(matchId, current.activeRecipientId, msg.sender);
    }
}
