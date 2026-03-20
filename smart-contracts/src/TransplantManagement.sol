// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/* TransplantManagement (v2.2) */
contract TransplantManagement {
    // ─────────────────────────────────────────────────────────────────────────────
    // Roles / Governance
    // ─────────────────────────────────────────────────────────────────────────────
    address public regulator;

    mapping(address => bool) public registeredHospitals;
    mapping(address => bool) public registeredMedicalTeams;
    mapping(address => bool) public authorizedLLMs;
    mapping(address => bool) public registeredEthicalCommittee;

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

    modifier onlyLLM() {
        require(authorizedLLMs[msg.sender], "Only authorized LLM");
        _;
    }

    modifier onlyEthicalCommittee() {
        require(registeredEthicalCommittee[msg.sender], "Only ethical committee");
        _;
    }

    modifier onlyHospitalOrMedicalTeam() {
        require(
            registeredHospitals[msg.sender] || registeredMedicalTeams[msg.sender],
            "Only hospital or medical team"
        );
        _;
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Core Data
    // ─────────────────────────────────────────────────────────────────────────────
    struct Donor {
        uint256 donorId;
        address donorAddress;
        string bloodType;
        string hlaTyping;
        string organType;
        string ipfsHash; // CID for donor medical record
        bool registered;
        bool ethicalApproved;
    }

    struct Recipient {
        uint256 recipientId;
        address recipientAddress;
        string bloodType;
        string hlaTyping;
        string organType;
        string ipfsHash; // CID for recipient medical record
        bool registered;
        bool matched; // used as a real guard in v2.2
        bool ethicalApproved;
    }

    struct Match {
        uint256 matchId;
        uint256 donorId;

        uint256 primaryRecipientId;
        uint256 backupRecipientId;

        uint256 activeRecipientId;
        bool backupPromoted;

        address matchedByLLM;
        string matchCID; // CID for LLM output (rationale + metadata)

        bool medicalApproved;
        bool hospitalApproved;
        bool donorApproved;
        bool activeRecipientApproved;
        bool ethicalCommitteeApproved;

        bool finalized;
    }

    mapping(uint256 => Donor) public donors;
    mapping(uint256 => Recipient) public recipients;
    mapping(uint256 => Match) public matches;

    mapping(address => uint256) public registeredDonorAddresses;     // donorAddress -> donorId
    mapping(address => uint256) public registeredRecipientAddresses; // recipientAddress -> recipientId

    // One open match per donor
    mapping(uint256 => bool) public donorHasOpenMatch;

    uint256 public donorCounter;
    uint256 public recipientCounter;
    uint256 public matchCounter;

    // ─────────────────────────────────────────────────────────────────────────────
    // Events
    // ─────────────────────────────────────────────────────────────────────────────
    event RegulatorChanged(address indexed oldRegulator, address indexed newRegulator);

    event HospitalRegistered(address indexed hospital);
    event MedicalTeamRegistered(address indexed medicalTeam);
    event EthicalCommitteeMemberRegistered(address indexed committeeMember);
    event LLMRegistered(address indexed llmAddress);

    event DonorAddressRegistered(address indexed donorAddress, uint256 indexed donorId);
    event RecipientAddressRegistered(address indexed recipientAddress, uint256 indexed recipientId);

    event DonorRegistered(uint256 indexed donorId, address indexed donorAddress, string organType, string ipfsHash);
    event RecipientRegistered(uint256 indexed recipientId, address indexed recipientAddress, string organType, string ipfsHash);

    event EthicalApprovalGranted(uint256 indexed id, string entityType);

    event MatchCreated(
        uint256 indexed matchId,
        uint256 indexed donorId,
        uint256 indexed primaryRecipientId,
        uint256 backupRecipientId,
        string matchCID
    );

    event BackupRecipientPromoted(
        uint256 indexed matchId,
        uint256 indexed oldActiveRecipientId,
        uint256 indexed newActiveRecipientId,
        address promotedBy
    );

    event ApprovalGranted(uint256 indexed matchId, string approvedBy);
    event MatchFinalized(uint256 indexed matchId, bool approved);

    // ─────────────────────────────────────────────────────────────────────────────
    // Constructor
    // ─────────────────────────────────────────────────────────────────────────────
    constructor(address _regulator) {
        require(_regulator != address(0), "Invalid regulator");
        regulator = _regulator;
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Internal helpers
    // ─────────────────────────────────────────────────────────────────────────────
    function _requireMatchExists(uint256 matchId) internal view {
        require(matches[matchId].matchId != 0, "Match does not exist");
    }

    function _requireNotFinalized(uint256 matchId) internal view {
        require(!matches[matchId].finalized, "Match finalized");
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Admin / Registration (Regulator)
    // ─────────────────────────────────────────────────────────────────────────────
    function changeRegulator(address newRegulator) external onlyRegulator {
        require(newRegulator != address(0), "Invalid regulator");
        address old = regulator;
        regulator = newRegulator;
        emit RegulatorChanged(old, newRegulator);
    }

    function registerHospital(address hospital) external onlyRegulator {
        require(hospital != address(0), "Invalid hospital");
        registeredHospitals[hospital] = true;
        emit HospitalRegistered(hospital);
    }

    function registerMedicalTeam(address medicalTeam) external onlyRegulator {
        require(medicalTeam != address(0), "Invalid medical team");
        registeredMedicalTeams[medicalTeam] = true;
        emit MedicalTeamRegistered(medicalTeam);
    }

    function registerLLM(address llm) external onlyRegulator {
        require(llm != address(0), "Invalid LLM");
        authorizedLLMs[llm] = true;
        emit LLMRegistered(llm);
    }

    function registerEthicalCommittee(address committeeMember) external onlyRegulator {
        require(committeeMember != address(0), "Invalid committee member");
        registeredEthicalCommittee[committeeMember] = true;
        emit EthicalCommitteeMemberRegistered(committeeMember);
    }

    function registerDonorAddress(address donorAddress) external onlyRegulator {
        require(donorAddress != address(0), "Invalid donor address");
        require(registeredDonorAddresses[donorAddress] == 0, "Donor already pre-registered");
        donorCounter += 1;
        registeredDonorAddresses[donorAddress] = donorCounter;
        emit DonorAddressRegistered(donorAddress, donorCounter);
    }

    function registerRecipientAddress(address recipientAddress) external onlyRegulator {
        require(recipientAddress != address(0), "Invalid recipient address");
        require(registeredRecipientAddresses[recipientAddress] == 0, "Recipient already pre-registered");
        recipientCounter += 1;
        registeredRecipientAddresses[recipientAddress] = recipientCounter;
        emit RecipientAddressRegistered(recipientAddress, recipientCounter);
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Hospital: register donor/recipient medical summary + CID
    // ─────────────────────────────────────────────────────────────────────────────
    function registerDonor(
        address donorAddress,
        string calldata bloodType,
        string calldata hlaTyping,
        string calldata organType,
        string calldata ipfsHash
    ) external onlyHospital {
        uint256 donorId = registeredDonorAddresses[donorAddress];
        require(donorId != 0, "Donor not pre-registered");
        require(!donors[donorId].registered, "Donor already registered");
        require(bytes(ipfsHash).length > 0, "Empty donor CID");

        Donor storage d = donors[donorId];
        d.donorId = donorId;
        d.donorAddress = donorAddress;
        d.bloodType = bloodType;
        d.hlaTyping = hlaTyping;
        d.organType = organType;
        d.ipfsHash = ipfsHash;
        d.registered = true;
        d.ethicalApproved = false;

        emit DonorRegistered(donorId, donorAddress, organType, ipfsHash);
    }

    function registerRecipient(
        address recipientAddress,
        string calldata bloodType,
        string calldata hlaTyping,
        string calldata organType,
        string calldata ipfsHash
    ) external onlyHospital {
        uint256 recipientId = registeredRecipientAddresses[recipientAddress];
        require(recipientId != 0, "Recipient not pre-registered");
        require(!recipients[recipientId].registered, "Recipient already registered");
        require(bytes(ipfsHash).length > 0, "Empty recipient CID");

        Recipient storage r = recipients[recipientId];
        r.recipientId = recipientId;
        r.recipientAddress = recipientAddress;
        r.bloodType = bloodType;
        r.hlaTyping = hlaTyping;
        r.organType = organType;
        r.ipfsHash = ipfsHash;
        r.registered = true;
        r.matched = false;
        r.ethicalApproved = false;

        emit RecipientRegistered(recipientId, recipientAddress, organType, ipfsHash);
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Ethical Committee: pre-match eligibility approvals
    // ─────────────────────────────────────────────────────────────────────────────
    function approveDonorEthicalCommittee(uint256 donorId) external onlyEthicalCommittee {
        require(donors[donorId].registered, "Donor not registered");
        require(!donors[donorId].ethicalApproved, "Donor already approved");
        donors[donorId].ethicalApproved = true;
        emit EthicalApprovalGranted(donorId, "Donor");
    }

    function approveRecipientEthicalCommittee(uint256 recipientId) external onlyEthicalCommittee {
        require(recipients[recipientId].registered, "Recipient not registered");
        require(!recipients[recipientId].ethicalApproved, "Recipient already approved");
        recipients[recipientId].ethicalApproved = true;
        emit EthicalApprovalGranted(recipientId, "Recipient");
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // LLM: create match (primary + backup + matchCID)
    // ─────────────────────────────────────────────────────────────────────────────
    function createMatch(
        uint256 donorId,
        uint256 primaryRecipientId,
        uint256 backupRecipientId,
        string calldata matchCID
    ) external onlyLLM {
        require(donors[donorId].registered, "Donor not registered");
        require(donors[donorId].ethicalApproved, "Donor must be ethically approved");
        require(!donorHasOpenMatch[donorId], "Donor already has open match");

        require(recipients[primaryRecipientId].registered, "Primary not registered");
        require(recipients[primaryRecipientId].ethicalApproved, "Primary must be ethically approved");
        require(!recipients[primaryRecipientId].matched, "Primary already matched");

        require(recipients[backupRecipientId].registered, "Backup not registered");
        require(recipients[backupRecipientId].ethicalApproved, "Backup must be ethically approved");
        require(!recipients[backupRecipientId].matched, "Backup already matched");

        require(primaryRecipientId != 0 && backupRecipientId != 0, "Invalid recipient ids");
        require(primaryRecipientId != backupRecipientId, "Primary and backup must differ");
        require(bytes(matchCID).length > 0, "Empty matchCID");

        matchCounter += 1;

        Match storage m = matches[matchCounter];
        m.matchId = matchCounter;
        m.donorId = donorId;

        m.primaryRecipientId = primaryRecipientId;
        m.backupRecipientId = backupRecipientId;

        m.activeRecipientId = primaryRecipientId;
        m.backupPromoted = false;

        m.matchedByLLM = msg.sender;
        m.matchCID = matchCID;

        m.medicalApproved = false;
        m.hospitalApproved = false;
        m.donorApproved = false;
        m.activeRecipientApproved = false;
        m.ethicalCommitteeApproved = false;
        m.finalized = false;

        donorHasOpenMatch[donorId] = true;
        recipients[primaryRecipientId].matched = true;
        recipients[backupRecipientId].matched = true;

        emit MatchCreated(matchCounter, donorId, primaryRecipientId, backupRecipientId, matchCID);
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Backup promotion: switch active recipient to backup
    // ─────────────────────────────────────────────────────────────────────────────
    function promoteBackupRecipient(uint256 matchId) external onlyHospitalOrMedicalTeam {
        _requireMatchExists(matchId);
        _requireNotFinalized(matchId);

        Match storage m = matches[matchId];

        require(!m.backupPromoted, "Backup already promoted");
        require(m.activeRecipientId == m.primaryRecipientId, "Active is not primary");

        // Promotion must happen before recipient approval and before final ethical approval
        require(!m.activeRecipientApproved, "Recipient already approved");
        require(!m.ethicalCommitteeApproved, "Final ethics already approved");

        uint256 oldActive = m.activeRecipientId;
        m.activeRecipientId = m.backupRecipientId;
        m.backupPromoted = true;

        // Explicitly keep recipient approval false because active recipient changed
        m.activeRecipientApproved = false;

        emit BackupRecipientPromoted(matchId, oldActive, m.activeRecipientId, msg.sender);
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Approvals (strict sequential order, no duplicates, no post-finalization)
    // ─────────────────────────────────────────────────────────────────────────────
    function approveMedicalTeam(uint256 matchId) external onlyMedicalTeam {
        _requireMatchExists(matchId);
        _requireNotFinalized(matchId);

        Match storage m = matches[matchId];
        require(!m.medicalApproved, "Medical already approved");

        m.medicalApproved = true;
        emit ApprovalGranted(matchId, "Medical Team");
    }

    function approveHospital(uint256 matchId) external onlyHospital {
        _requireMatchExists(matchId);
        _requireNotFinalized(matchId);

        Match storage m = matches[matchId];
        require(m.medicalApproved, "Medical approval required first");
        require(!m.hospitalApproved, "Hospital already approved");

        m.hospitalApproved = true;
        emit ApprovalGranted(matchId, "Hospital");
    }

    function approveDonor(uint256 matchId) external {
        _requireMatchExists(matchId);
        _requireNotFinalized(matchId);

        Match storage m = matches[matchId];
        require(m.hospitalApproved, "Hospital approval required first");
        require(!m.donorApproved, "Donor already approved");

        uint256 donorId = m.donorId;
        require(msg.sender == donors[donorId].donorAddress, "Only donor can approve");

        m.donorApproved = true;
        emit ApprovalGranted(matchId, "Donor");
    }

    function approveRecipient(uint256 matchId) external {
        _requireMatchExists(matchId);
        _requireNotFinalized(matchId);

        Match storage m = matches[matchId];
        require(m.donorApproved, "Donor approval required first");
        require(!m.activeRecipientApproved, "Recipient already approved");

        uint256 activeRid = m.activeRecipientId;
        require(activeRid != 0, "No active recipient");
        require(msg.sender == recipients[activeRid].recipientAddress, "Only active recipient can approve");

        m.activeRecipientApproved = true;
        emit ApprovalGranted(matchId, "Recipient");
    }

    function approveFinalTransplant(uint256 matchId) external onlyEthicalCommittee {
        _requireMatchExists(matchId);
        _requireNotFinalized(matchId);

        Match storage m = matches[matchId];
        require(m.activeRecipientApproved, "Recipient approval required first");
        require(!m.ethicalCommitteeApproved, "Ethics already approved");

        m.ethicalCommitteeApproved = true;
        emit ApprovalGranted(matchId, "Ethical Committee");
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Final decision & finalize helper
    // ─────────────────────────────────────────────────────────────────────────────
    function isTransplantApproved(uint256 matchId) public view returns (bool) {
        Match memory m = matches[matchId];
        return (
            m.medicalApproved &&
            m.hospitalApproved &&
            m.donorApproved &&
            m.activeRecipientApproved &&
            m.ethicalCommitteeApproved
        );
    }

    function finalizeMatch(uint256 matchId) external {
        _requireMatchExists(matchId);
        require(!matches[matchId].finalized, "Already finalized");
        require(isTransplantApproved(matchId), "Transplant not fully approved");

        Match storage m = matches[matchId];
        m.finalized = true;
        donorHasOpenMatch[m.donorId] = false;

        // Release the reserved, non-active recipient after the match is finalized.
        if (m.activeRecipientId == m.primaryRecipientId) {
            recipients[m.backupRecipientId].matched = false;
        } else {
            recipients[m.primaryRecipientId].matched = false;
        }

        emit MatchFinalized(matchId, true);
    }
}
