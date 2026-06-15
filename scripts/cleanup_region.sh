#!/usr/bin/env bash
# Delete all RAG Document AWS resources for this project.
#
# ACCOUNT_BASELINE_REGIONS lists every region that is enabled by default in
# this AWS account (verified in the console: ap-northeast-1, ap-south-2,
# eu-north-1).  These are always scanned.  Pass --region to add the pipeline
# region (e.g. us-east-1) when it differs from the baseline set.
#
# Flags:
#   --audit        Read-only: enumerate every resource in all regions by type.
#   --dry-run      Show what would be deleted without making any changes.
#   --include-iam  Also delete IAM roles + GitHub OIDC provider (global).
#   --include-vpc  Delete billable VPC resources (NAT GW, interface endpoints,
#                  unattached EIPs) and custom (non-default) VPCs.
#   --nuke         FULL WIPE. Implies --include-iam and --include-vpc.
#                  Additionally deletes: EC2 instances, ALL EIPs (attached too),
#                  ALL VPCs including defaults with every subnet / IGW / NACL /
#                  route table / DHCP option set, and the Terraform state bucket.
#                  Nothing in the targeted regions will survive.
#   --yes          Skip confirmation prompt (required with --nuke in CI).
#
# Usage:
#   ./scripts/cleanup_region.sh --audit
#   ./scripts/cleanup_region.sh --region us-east-1 --audit
#   ./scripts/cleanup_region.sh --dry-run
#   ./scripts/cleanup_region.sh --region us-east-1 --dry-run
#   ./scripts/cleanup_region.sh --region us-east-1 --include-vpc --include-iam
#   ./scripts/cleanup_region.sh --nuke --yes

set -euo pipefail

# ── Argument parsing ──────────────────────────────────────────────────────────

PIPELINE_REGION=""   # passed via --region (the region CI/CD deployed to)
PROJECT="rag-document-aws"
DRY_RUN=false
INCLUDE_IAM=false
INCLUDE_VPC=false
NUKE=false
AUDIT=false
AUTO_YES=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --region)      PIPELINE_REGION="$2"; shift 2 ;;
    --project)     PROJECT="$2";         shift 2 ;;
    --dry-run)     DRY_RUN=true;         shift   ;;
    --include-iam) INCLUDE_IAM=true;     shift   ;;
    --include-vpc) INCLUDE_VPC=true;     shift   ;;
    --nuke)        NUKE=true; INCLUDE_IAM=true; INCLUDE_VPC=true; shift ;;
    --audit)       AUDIT=true;           shift   ;;
    --yes)         AUTO_YES=true;        shift   ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

# ── Helpers ───────────────────────────────────────────────────────────────────

run() {
  if $DRY_RUN; then
    echo "    [DRY-RUN] $*"
  else
    "$@"
  fi
}

section() { echo; echo "══ $1"; }
info()    { echo "  → $*"; }
warn()    { echo "  ⚠  $*"; }
ok()      { echo "  ✓ $*"; }

# ── Resolve regions to scan ───────────────────────────────────────────────────

# Regions enabled by default in this AWS account (verified in the console).
# These are always included in every scan run.
declare -a ACCOUNT_BASELINE_REGIONS=("ap-northeast-1" "ap-south-2" "eu-north-1")

# Build the final scan list: baseline + pipeline region (deduped, order preserved)
declare -a REGIONS_TO_SCAN=("${ACCOUNT_BASELINE_REGIONS[@]}")
if [[ -n "$PIPELINE_REGION" ]]; then
  already=false
  for r in "${ACCOUNT_BASELINE_REGIONS[@]}"; do
    [[ "$r" == "$PIPELINE_REGION" ]] && already=true && break
  done
  $already || REGIONS_TO_SCAN+=("$PIPELINE_REGION")
fi

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text 2>/dev/null || echo "unknown")

section "Account & region summary ════════════════════════════════"
echo "  Account ID          : $ACCOUNT_ID"
echo "  Baseline regions    : ${ACCOUNT_BASELINE_REGIONS[*]}"
echo "                        (default-enabled in this account, always scanned)"
if [[ -n "$PIPELINE_REGION" ]]; then
  already=false
  for r in "${ACCOUNT_BASELINE_REGIONS[@]}"; do
    [[ "$r" == "$PIPELINE_REGION" ]] && already=true && break
  done
  if $already; then
    echo "  Pipeline region     : $PIPELINE_REGION  (already in baseline — no extra scan)"
  else
    echo "  Pipeline region     : $PIPELINE_REGION  (added to scan list)"
  fi
else
  echo "  Pipeline region     : (not specified — pass --region <region> to add one)"
fi
echo "  Regions to scan     : ${REGIONS_TO_SCAN[*]}"
$DRY_RUN && echo "  Mode                : DRY-RUN (no changes will be made)"
$AUDIT   && echo "  Mode                : AUDIT   (read-only, full resource enumeration)"
if $NUKE; then
  echo "  Mode                : NUKE"
  echo
  echo "  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
  echo "  !!  NUKE MODE — EVERY resource in the listed regions will   !!"
  echo "  !!  be permanently destroyed, including default VPCs,       !!"
  echo "  !!  all subnets, EC2 instances, EIPs, DHCP option sets,     !!"
  echo "  !!  and the Terraform state bucket. This cannot be undone.  !!"
  echo "  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
  echo
  if ! $AUTO_YES; then
    read -r -p "  Type 'nuke' to confirm full wipe: " nuke_confirm
    [[ "$nuke_confirm" == "nuke" ]] || { echo "Aborted."; exit 0; }
  fi
fi

# ── Audit function — full resource enumeration per region ─────────────────────
# Uses EC2 + Resource Groups Tagging API to list every resource type present,
# grouped by service, with counts.  Marks known AWS-default (free) resources.
# This mirrors what the AWS Console "Resource Groups" view shows.
audit_region() {
  local region="$1"
  section "AUDIT — $region ══════════════════════════════════════════"

  # ── EC2 / VPC resources (incl. AWS defaults) ─────────────────────────────

  local vpcs subnets sgs rtbs nacls igws eips dhcp
  vpcs=$(aws ec2 describe-vpcs --region "$region" \
    --query "Vpcs[].[VpcId,IsDefault]" --output text 2>/dev/null || true)
  subnets=$(aws ec2 describe-subnets --region "$region" \
    --query "Subnets[].[SubnetId,DefaultForAz,VpcId]" --output text 2>/dev/null || true)
  sgs=$(aws ec2 describe-security-groups --region "$region" \
    --query "SecurityGroups[].[GroupId,GroupName,VpcId]" --output text 2>/dev/null || true)
  rtbs=$(aws ec2 describe-route-tables --region "$region" \
    --query "RouteTables[].[RouteTableId,VpcId,Associations[0].Main]" --output text 2>/dev/null || true)
  nacls=$(aws ec2 describe-network-acls --region "$region" \
    --query "NetworkAcls[].[NetworkAclId,IsDefault,VpcId]" --output text 2>/dev/null || true)
  igws=$(aws ec2 describe-internet-gateways --region "$region" \
    --query "InternetGateways[].[InternetGatewayId,Attachments[0].VpcId]" --output text 2>/dev/null || true)
  eips=$(aws ec2 describe-addresses --region "$region" \
    --query "Addresses[].[AllocationId,AssociationId,PublicIp]" --output text 2>/dev/null || true)
  dhcp=$(aws ec2 describe-dhcp-options --region "$region" \
    --query "DhcpOptions[].DhcpOptionsId" --output text 2>/dev/null || true)

  echo
  echo "  EC2 / VPC resources:"
  echo "  ┌─────────────────────────────────────────────────────────────────"

  # VPCs
  local vpc_default_count=0 vpc_custom_count=0
  while IFS=$'\t' read -r vpc_id is_default; do
    [[ -z "$vpc_id" ]] && continue
    if [[ "$is_default" == "True" ]]; then
      echo "  │  VPC (DEFAULT) : $vpc_id  — free, AWS-managed"
      (( vpc_default_count++ )) || true
    else
      echo "  │  VPC (CUSTOM)  : $vpc_id  *** custom — check if needed"
      (( vpc_custom_count++ )) || true
    fi
  done <<< "$vpcs"

  # Subnets
  local sn_default=0 sn_custom=0
  while IFS=$'\t' read -r sn_id is_default vpc_id; do
    [[ -z "$sn_id" ]] && continue
    if [[ "$is_default" == "True" ]]; then
      (( sn_default++ )) || true
    else
      echo "  │  Subnet (CUSTOM): $sn_id  vpc=$vpc_id  *** custom"
      (( sn_custom++ )) || true
    fi
  done <<< "$subnets"
  [[ $sn_default -gt 0 ]] && echo "  │  Subnets (default) : $sn_default × default subnets  — free, AWS-managed"

  # Security groups
  local sg_default=0 sg_custom=0
  while IFS=$'\t' read -r sg_id sg_name vpc_id; do
    [[ -z "$sg_id" ]] && continue
    if [[ "$sg_name" == "default" ]]; then
      (( sg_default++ )) || true
    else
      echo "  │  SG (CUSTOM)   : $sg_id  \"$sg_name\"  vpc=$vpc_id  *** custom"
      (( sg_custom++ )) || true
    fi
  done <<< "$sgs"
  [[ $sg_default -gt 0 ]] && echo "  │  Security groups (default) : $sg_default × default SG  — free, cannot be deleted"

  # Route tables
  local rtb_main=0 rtb_custom=0
  while IFS=$'\t' read -r rtb_id vpc_id is_main; do
    [[ -z "$rtb_id" ]] && continue
    if [[ "$is_main" == "True" ]]; then
      (( rtb_main++ )) || true
    else
      echo "  │  Route table (CUSTOM): $rtb_id  vpc=$vpc_id  *** custom"
      (( rtb_custom++ )) || true
    fi
  done <<< "$rtbs"
  [[ $rtb_main -gt 0 ]] && echo "  │  Route tables (main)  : $rtb_main × main route table  — free, AWS-managed"

  # NACLs
  local nacl_default=0 nacl_custom=0
  while IFS=$'\t' read -r nacl_id is_default vpc_id; do
    [[ -z "$nacl_id" ]] && continue
    if [[ "$is_default" == "True" ]]; then
      (( nacl_default++ )) || true
    else
      echo "  │  NACL (CUSTOM) : $nacl_id  vpc=$vpc_id  *** custom"
      (( nacl_custom++ )) || true
    fi
  done <<< "$nacls"
  [[ $nacl_default -gt 0 ]] && echo "  │  NACLs (default) : $nacl_default × default NACL  — free, AWS-managed"

  # Internet Gateways
  while IFS=$'\t' read -r igw_id attached_vpc; do
    [[ -z "$igw_id" ]] && continue
    echo "  │  Internet GW   : $igw_id  attached=$attached_vpc  — free (no hourly charge)"
  done <<< "$igws"

  # DHCP option sets
  local dhcp_count
  dhcp_count=$(echo "$dhcp" | grep -c '\S' || true)
  [[ $dhcp_count -gt 0 ]] && echo "  │  DHCP options  : $dhcp_count set(s)  — free, AWS-managed"

  # Elastic IPs
  local eip_attached=0 eip_unattached=0
  while IFS=$'\t' read -r alloc_id assoc_id pub_ip; do
    [[ -z "$alloc_id" ]] && continue
    if [[ -z "$assoc_id" || "$assoc_id" == "None" ]]; then
      echo "  │  EIP (UNATTACHED): $alloc_id  $pub_ip  *** BILLABLE ~\$0.005/hr"
      (( eip_unattached++ )) || true
    else
      echo "  │  EIP (attached): $alloc_id  $pub_ip  — in use"
      (( eip_attached++ )) || true
    fi
  done <<< "$eips"

  # NAT Gateways
  local nats
  nats=$(aws ec2 describe-nat-gateways --region "$region" \
    --filter Name=state,Values=available,pending \
    --query "NatGateways[].[NatGatewayId,VpcId,SubnetId]" --output text 2>/dev/null || true)
  while IFS=$'\t' read -r nat_id vpc_id sn_id; do
    [[ -z "$nat_id" ]] && continue
    echo "  │  NAT Gateway   : $nat_id  vpc=$vpc_id  *** BILLABLE ~\$0.045/hr"
  done <<< "$nats"

  # Interface VPC Endpoints
  local endpoints
  endpoints=$(aws ec2 describe-vpc-endpoints --region "$region" \
    --filters Name=vpc-endpoint-type,Values=Interface Name=vpc-endpoint-state,Values=available,pending \
    --query "VpcEndpoints[].[VpcEndpointId,ServiceName,VpcId]" --output text 2>/dev/null || true)
  while IFS=$'\t' read -r ep_id svc vpc_id; do
    [[ -z "$ep_id" ]] && continue
    echo "  │  Interface endpoint: $ep_id  svc=$svc  *** BILLABLE ~\$0.01/hr/AZ"
  done <<< "$endpoints"

  echo "  └─────────────────────────────────────────────────────────────────"

  # ── All tagged resources via Resource Groups Tagging API ─────────────────
  # This catches every service (Bedrock, S3, OpenSearch, CloudWatch, etc.)
  # that was created with tags, grouped by resource type.

  echo
  echo "  All tagged resources (by type):"
  echo "  ┌─────────────────────────────────────────────────────────────────"

  local tagged_json
  tagged_json=$(aws resourcegroupstaggingapi get-resources \
    --region "$region" \
    --query "ResourceTagMappingList[].ResourceARN" \
    --output text 2>/dev/null || true)

  if [[ -z "$tagged_json" ]]; then
    echo "  │  (no tagged resources found — untagged resources not shown here)"
  else
    # Extract service:type from ARN (arn:aws:SERVICE::ACCOUNT:TYPE/id)
    declare -A type_counts
    while read -r arn; do
      [[ -z "$arn" ]] && continue
      # arn:aws:service:region:account:type/resource  or  arn:aws:service:::type/resource
      local svc_type
      svc_type=$(echo "$arn" | awk -F: '{
        svc=$4; rest=$NF
        # rest is "type/resource" or just "resource"
        n=split(rest,a,"/")
        if(n>1) { print svc ":" a[1] }
        else    { print svc ":" rest }
      }')
      type_counts["$svc_type"]=$(( ${type_counts["$svc_type"]:-0} + 1 ))
    done <<< "$(echo "$tagged_json" | tr '\t' '\n')"

    for t in $(echo "${!type_counts[@]}" | tr ' ' '\n' | sort); do
      printf "  │  %-45s : %d\n" "$t" "${type_counts[$t]}"
    done
  fi

  echo "  └─────────────────────────────────────────────────────────────────"

  # ── Untagged resource note ────────────────────────────────────────────────
  echo
  echo "  NOTE: Default VPC resources (VPC, subnets, route tables, NACLs,"
  echo "        security groups, DHCP options, IGW) are untagged by AWS and"
  echo "        do NOT appear in the tagged list above — they are free and"
  echo "        shown separately in the EC2/VPC section."
}

# ── Short-circuit: audit mode exits after reporting ───────────────────────────
if $AUDIT; then
  for r in "${REGIONS_TO_SCAN[@]}"; do
    audit_region "$r"
  done
  echo
  echo "Audit complete. No resources were modified."
  echo "Re-run without --audit to proceed with cleanup."
  exit 0
fi

# ── Per-region inventory function ─────────────────────────────────────────────

# Populates global arrays: REGION_S3 REGION_AOSS_COLS REGION_AOSS_ENC
#                          REGION_AOSS_NET REGION_AOSS_ACCESS REGION_CW
scan_region() {
  local region="$1"

  section "Inventory — $region ════════════════════════════════════"

  # S3 (global API; filter by bucket location)
  # nuke=true  → include tfstate bucket too
  # nuke=false → project-prefixed + tfstate buckets only (tfstate skipped at deletion time)
  REGION_S3=()
  local all_buckets
  all_buckets=$(aws s3api list-buckets --query "Buckets[].Name" --output text 2>/dev/null || true)
  for bucket in $all_buckets; do
    if [[ "$bucket" == ${PROJECT}* ]] || [[ "$bucket" == *"-tfstate-"* ]]; then
      local loc
      loc=$(aws s3api get-bucket-location --bucket "$bucket" \
            --query "LocationConstraint" --output text 2>/dev/null || echo "unknown")
      [[ "$loc" == "None" ]] && loc="us-east-1"   # us-east-1 returns None
      if [[ "$loc" == "$region" ]]; then
        REGION_S3+=("$bucket")
        if [[ "$bucket" == *"-tfstate-"* ]]; then
          $NUKE && info "S3 bucket (tfstate): $bucket  [included in --nuke]" \
                || info "S3 bucket (tfstate): $bucket  [skipped unless --nuke]"
        else
          info "S3 bucket          : $bucket"
        fi
      fi
    fi
  done

  # OpenSearch Serverless
  REGION_AOSS_COLS=$(aws opensearchserverless list-collections --region "$region" \
    --query "collectionSummaries[?starts_with(name,'rag-')].name" \
    --output text 2>/dev/null || true)

  REGION_AOSS_ENC=$(aws opensearchserverless list-security-policies --type encryption \
    --region "$region" \
    --query "securityPolicySummaries[?starts_with(name,'${PROJECT}')].name" \
    --output text 2>/dev/null || true)

  REGION_AOSS_NET=$(aws opensearchserverless list-security-policies --type network \
    --region "$region" \
    --query "securityPolicySummaries[?starts_with(name,'${PROJECT}')].name" \
    --output text 2>/dev/null || true)

  REGION_AOSS_ACCESS=$(aws opensearchserverless list-access-policies --type data \
    --region "$region" \
    --query "accessPolicySummaries[?starts_with(name,'${PROJECT}')].name" \
    --output text 2>/dev/null || true)

  for c in $REGION_AOSS_COLS;   do info "OpenSearch collection  : $c  [ACTIVE — billed by OCU-hour]"; done
  for p in $REGION_AOSS_ENC;    do info "OpenSearch enc policy  : $p"; done
  for p in $REGION_AOSS_NET;    do info "OpenSearch net policy  : $p"; done
  for p in $REGION_AOSS_ACCESS; do info "OpenSearch access pol  : $p"; done

  # CloudWatch log groups
  REGION_CW=$(aws logs describe-log-groups --region "$region" \
    --log-group-name-prefix "/aws/bedrock" \
    --query "logGroups[?contains(logGroupName,'${PROJECT}') || contains(logGroupName,'rag-document')].logGroupName" \
    --output text 2>/dev/null || true)

  for lg in $REGION_CW; do info "CloudWatch log group   : $lg"; done

  # VPC resources
  scan_vpc_region "$region"

  local found=false
  [[ ${#REGION_S3[@]} -gt 0 ]]        && found=true
  [[ -n "$REGION_AOSS_COLS" ]]        && found=true
  [[ -n "$REGION_AOSS_ENC" ]]         && found=true
  [[ -n "$REGION_AOSS_NET" ]]         && found=true
  [[ -n "$REGION_AOSS_ACCESS" ]]      && found=true
  [[ -n "$REGION_CW" ]]               && found=true
  [[ -n "$REGION_VPC_BILLABLE" ]]     && found=true
  [[ -n "$REGION_VPC_CUSTOM" ]]       && found=true
  $found || echo "  (nothing found in $region)"
}

# ── VPC inventory function ────────────────────────────────────────────────────
# Populates globals:
#   REGION_VPC_DEFAULT   — default VPC ids (free, always skipped)
#   REGION_VPC_CUSTOM    — non-default VPC ids (deleted with --include-vpc)
#   REGION_NAT_GW        — NAT gateway ids          [BILLABLE]
#   REGION_VPC_ENDPOINTS — interface endpoint ids   [BILLABLE]
#   REGION_EIP_UNATTACHED— unattached Elastic IP allocation ids [BILLABLE]
#   REGION_VPC_BILLABLE  — combined non-empty marker
scan_vpc_region() {
  local region="$1"

  echo "  [VPC]"

  # VPCs — split default from custom
  REGION_VPC_DEFAULT=$(aws ec2 describe-vpcs --region "$region" \
    --filters Name=isDefault,Values=true \
    --query "Vpcs[].VpcId" --output text 2>/dev/null || true)

  REGION_VPC_CUSTOM=$(aws ec2 describe-vpcs --region "$region" \
    --filters Name=isDefault,Values=false \
    --query "Vpcs[].VpcId" --output text 2>/dev/null || true)

  for v in $REGION_VPC_DEFAULT; do
    echo "    DEFAULT VPC        : $v  (free — AWS-managed, skipped by default)"
    # Report what's inside the default VPC so the user can see it's clean
    local def_subnets def_sgs def_rtbs
    def_subnets=$(aws ec2 describe-subnets --region "$region" \
      --filters Name=vpc-id,Values="$v" Name=defaultForAz,Values=true \
      --query "Subnets[].SubnetId" --output text 2>/dev/null || true)
    def_sgs=$(aws ec2 describe-security-groups --region "$region" \
      --filters Name=vpc-id,Values="$v" Name=group-name,Values=default \
      --query "SecurityGroups[].GroupId" --output text 2>/dev/null || true)
    def_rtbs=$(aws ec2 describe-route-tables --region "$region" \
      --filters Name=vpc-id,Values="$v" \
      --query "RouteTables[].RouteTableId" --output text 2>/dev/null || true)
    [[ -n "$def_subnets" ]] && echo "      subnets (default) : $def_subnets  — free, skipped"
    [[ -n "$def_sgs"     ]] && echo "      security groups   : $def_sgs  — free, skipped"
    [[ -n "$def_rtbs"    ]] && echo "      route tables      : $def_rtbs  — free, skipped"
  done

  for v in $REGION_VPC_CUSTOM; do
    local cidr name_tag
    cidr=$(aws ec2 describe-vpcs --region "$region" --vpc-ids "$v" \
      --query "Vpcs[0].CidrBlock" --output text 2>/dev/null || true)
    name_tag=$(aws ec2 describe-vpcs --region "$region" --vpc-ids "$v" \
      --query "Vpcs[0].Tags[?Key=='Name'].Value | [0]" --output text 2>/dev/null || true)
    warn "CUSTOM VPC         : $v  cidr=$cidr name=${name_tag:-<none>}  (--include-vpc to delete)"
  done

  # Billable: NAT Gateways (charged per hour even when idle)
  REGION_NAT_GW=$(aws ec2 describe-nat-gateways --region "$region" \
    --filter Name=state,Values=available,pending \
    --query "NatGateways[].NatGatewayId" --output text 2>/dev/null || true)
  for n in $REGION_NAT_GW; do
    warn "NAT Gateway        : $n  [BILLABLE ~\$0.045/hr]  (--include-vpc to delete)"
  done

  # Billable: Interface VPC Endpoints (Gateway endpoints are free)
  REGION_VPC_ENDPOINTS=$(aws ec2 describe-vpc-endpoints --region "$region" \
    --filters Name=vpc-endpoint-type,Values=Interface Name=vpc-endpoint-state,Values=available,pending \
    --query "VpcEndpoints[].VpcEndpointId" --output text 2>/dev/null || true)
  for e in $REGION_VPC_ENDPOINTS; do
    warn "Interface endpoint : $e  [BILLABLE ~\$0.01/hr/AZ]  (--include-vpc to delete)"
  done

  # Billable: Elastic IPs not attached to anything
  REGION_EIP_UNATTACHED=$(aws ec2 describe-addresses --region "$region" \
    --query "Addresses[?AssociationId==null].AllocationId" \
    --output text 2>/dev/null || true)
  for e in $REGION_EIP_UNATTACHED; do
    warn "Unattached EIP     : $e  [BILLABLE ~\$0.005/hr]  (--include-vpc to delete)"
  done

  # Combined marker used by the caller to detect any VPC findings
  REGION_VPC_BILLABLE="${REGION_NAT_GW}${REGION_VPC_ENDPOINTS}${REGION_EIP_UNATTACHED}"
}

# ── Per-region deletion function ──────────────────────────────────────────────

delete_region() {
  local region="$1"

  # Re-scan to get fresh values for this region
  scan_region "$region"

  local s3_buckets=("${REGION_S3[@]+"${REGION_S3[@]}"}")
  local aoss_cols="$REGION_AOSS_COLS"
  local aoss_enc="$REGION_AOSS_ENC"
  local aoss_net="$REGION_AOSS_NET"
  local aoss_access="$REGION_AOSS_ACCESS"
  local cw_groups="$REGION_CW"

  # OpenSearch — collection first, then policies (collection refs the policies)
  if [[ -n "$aoss_cols" ]]; then
    section "Deleting OpenSearch — $region ════════════════════════"
    for c in $aoss_cols; do
      local col_id
      col_id=$(aws opensearchserverless list-collections --region "$region" \
        --query "collectionSummaries[?name=='$c'].id" --output text)
      info "Deleting collection: $c  (id: $col_id)"
      run aws opensearchserverless delete-collection --id "$col_id" --region "$region"
      ok "$c delete initiated (takes ~1 min)"
    done
    if ! $DRY_RUN; then
      echo "  Waiting 30 s for collection deletion to progress..."
      sleep 30
    fi
  fi

  for p in $aoss_access; do
    info "Deleting access policy: $p"
    run aws opensearchserverless delete-access-policy --type data --name "$p" --region "$region"
    ok "$p deleted"
  done
  for p in $aoss_net; do
    info "Deleting network policy: $p"
    run aws opensearchserverless delete-security-policy --type network --name "$p" --region "$region"
    ok "$p deleted"
  done
  for p in $aoss_enc; do
    info "Deleting encryption policy: $p"
    run aws opensearchserverless delete-security-policy --type encryption --name "$p" --region "$region"
    ok "$p deleted"
  done

  # CloudWatch
  if [[ -n "$cw_groups" ]]; then
    section "Deleting CloudWatch — $region ═══════════════════════"
    for lg in $cw_groups; do
      info "Deleting log group: $lg"
      run aws logs delete-log-group --log-group-name "$lg" --region "$region"
      ok "$lg deleted"
    done
  fi

  # VPC billable resources + custom VPCs (opt-in)
  if $INCLUDE_VPC; then
    delete_vpc_region "$region"
  fi

  # S3 — empty then delete (skip tfstate bucket)
  if [[ ${#s3_buckets[@]} -gt 0 ]]; then
    section "Deleting S3 — $region ═══════════════════════════════"
    for bucket in "${s3_buckets[@]}"; do
      if [[ "$bucket" == *"-tfstate-"* ]] && ! $NUKE; then
        warn "Skipping Terraform state bucket: $bucket  (pass --nuke to include it)"
        continue
      fi
      info "Emptying: $bucket"
      if ! $DRY_RUN; then
        # Remove versioned objects
        local versions
        versions=$(aws s3api list-object-versions --bucket "$bucket" \
          --query '{Objects: Versions[].{Key:Key,VersionId:VersionId}}' \
          --output json 2>/dev/null || echo '{"Objects":[]}')
        [[ "$versions" != '{"Objects":[]}' && "$versions" != '{"Objects":null}' ]] && \
          aws s3api delete-objects --bucket "$bucket" --delete "$versions" \
            --region "$region" > /dev/null 2>&1 || true
        # Remove delete markers
        local markers
        markers=$(aws s3api list-object-versions --bucket "$bucket" \
          --query '{Objects: DeleteMarkers[].{Key:Key,VersionId:VersionId}}' \
          --output json 2>/dev/null || echo '{"Objects":[]}')
        [[ "$markers" != '{"Objects":[]}' && "$markers" != '{"Objects":null}' ]] && \
          aws s3api delete-objects --bucket "$bucket" --delete "$markers" \
            --region "$region" > /dev/null 2>&1 || true
        # Catch-all for unversioned objects
        aws s3 rm "s3://$bucket" --recursive --region "$region" > /dev/null 2>&1 || true
      else
        echo "    [DRY-RUN] would empty s3://$bucket"
      fi
      info "Deleting bucket: $bucket"
      run aws s3api delete-bucket --bucket "$bucket" --region "$region"
      ok "$bucket deleted"
    done
  fi
}

# ── VPC deletion function ─────────────────────────────────────────────────────
# --include-vpc : billable resources + custom VPCs only
# --nuke        : everything above PLUS default VPCs, EC2 instances, all EIPs,
#                 DHCP option sets — leaves the region with zero VPC resources
delete_vpc_region() {
  local region="$1"
  section "VPC cleanup — $region ══════════════════════════════════"

  scan_vpc_region "$region" > /dev/null 2>&1 || true

  # ── Step 1: EC2 instances (nuke only — must go before subnets/VPCs) ────────
  if $NUKE; then
    local instances
    instances=$(aws ec2 describe-instances --region "$region" \
      --filters Name=instance-state-name,Values=pending,running,stopping,stopped \
      --query "Reservations[].Instances[].InstanceId" --output text 2>/dev/null || true)
    if [[ -n "$instances" ]]; then
      info "Terminating EC2 instances: $instances"
      run aws ec2 terminate-instances --instance-ids $instances --region "$region"
      if ! $DRY_RUN; then
        echo "  Waiting for instance termination..."
        aws ec2 wait instance-terminated --instance-ids $instances --region "$region"
        ok "All instances terminated"
      fi
    fi
  fi

  # ── Step 2: NAT Gateways (must go before subnets) ─────────────────────────
  for nat in $REGION_NAT_GW; do
    info "Deleting NAT Gateway: $nat"
    run aws ec2 delete-nat-gateway --nat-gateway-id "$nat" --region "$region"
    ok "$nat deletion initiated"
  done
  if [[ -n "$REGION_NAT_GW" ]] && ! $DRY_RUN; then
    echo "  Waiting 30 s for NAT Gateway deletion..."
    sleep 30
  fi

  # ── Step 3: Interface VPC Endpoints ───────────────────────────────────────
  for ep in $REGION_VPC_ENDPOINTS; do
    info "Deleting interface endpoint: $ep"
    run aws ec2 delete-vpc-endpoints --vpc-endpoint-ids "$ep" --region "$region"
    ok "$ep deleted"
  done

  # ── Step 4: Elastic IPs ───────────────────────────────────────────────────
  if $NUKE; then
    # Release ALL EIPs — disassociate first if still attached
    local all_eips
    all_eips=$(aws ec2 describe-addresses --region "$region" \
      --query "Addresses[].[AllocationId,AssociationId]" --output text 2>/dev/null || true)
    while IFS=$'\t' read -r alloc_id assoc_id; do
      [[ -z "$alloc_id" ]] && continue
      if [[ -n "$assoc_id" && "$assoc_id" != "None" ]]; then
        info "Disassociating EIP: $alloc_id  (assoc: $assoc_id)"
        run aws ec2 disassociate-address --association-id "$assoc_id" --region "$region"
      fi
      info "Releasing EIP: $alloc_id"
      run aws ec2 release-address --allocation-id "$alloc_id" --region "$region"
      ok "$alloc_id released"
    done <<< "$all_eips"
  else
    # --include-vpc: only unattached EIPs
    for alloc in $REGION_EIP_UNATTACHED; do
      info "Releasing unattached EIP: $alloc"
      run aws ec2 release-address --allocation-id "$alloc" --region "$region"
      ok "$alloc released"
    done
  fi

  # ── Step 5: Tear down VPCs ────────────────────────────────────────────────
  # nuke = all VPCs (default + custom); include-vpc = custom only
  local vpcs_to_delete
  if $NUKE; then
    vpcs_to_delete=$(aws ec2 describe-vpcs --region "$region" \
      --query "Vpcs[].VpcId" --output text 2>/dev/null || true)
  else
    vpcs_to_delete="$REGION_VPC_CUSTOM"
  fi

  for vpc in $vpcs_to_delete; do
    local is_default
    is_default=$(aws ec2 describe-vpcs --region "$region" --vpc-ids "$vpc" \
      --query "Vpcs[0].IsDefault" --output text 2>/dev/null || echo "False")
    info "Tearing down VPC: $vpc  (default=$is_default)"

    # All subnets
    local subnets
    subnets=$(aws ec2 describe-subnets --region "$region" \
      --filters Name=vpc-id,Values="$vpc" \
      --query "Subnets[].SubnetId" --output text 2>/dev/null || true)
    for sn in $subnets; do
      info "  Deleting subnet: $sn"
      run aws ec2 delete-subnet --subnet-id "$sn" --region "$region"
    done

    # Non-default security groups (the 'default' SG is deleted with the VPC)
    local sgs
    sgs=$(aws ec2 describe-security-groups --region "$region" \
      --filters Name=vpc-id,Values="$vpc" \
      --query "SecurityGroups[?GroupName!='default'].GroupId" --output text 2>/dev/null || true)
    for sg in $sgs; do
      info "  Deleting security group: $sg"
      run aws ec2 delete-security-group --group-id "$sg" --region "$region"
    done

    # Non-main route tables (main RTB is deleted with the VPC)
    local rtbs
    rtbs=$(aws ec2 describe-route-tables --region "$region" \
      --filters Name=vpc-id,Values="$vpc" \
      --query "RouteTables[?Associations[?Main==\`false\`] || length(Associations)==\`0\`].RouteTableId" \
      --output text 2>/dev/null || true)
    for rt in $rtbs; do
      info "  Deleting route table: $rt"
      run aws ec2 delete-route-table --route-table-id "$rt" --region "$region"
    done

    # Internet Gateway — must detach before deleting
    local igws
    igws=$(aws ec2 describe-internet-gateways --region "$region" \
      --filters Name=attachment.vpc-id,Values="$vpc" \
      --query "InternetGateways[].InternetGatewayId" --output text 2>/dev/null || true)
    for igw in $igws; do
      info "  Detaching IGW: $igw"
      run aws ec2 detach-internet-gateway --internet-gateway-id "$igw" --vpc-id "$vpc" --region "$region"
      info "  Deleting IGW: $igw"
      run aws ec2 delete-internet-gateway --internet-gateway-id "$igw" --region "$region"
    done

    # Delete VPC (takes default SG, default NACL, main RTB with it)
    info "  Deleting VPC: $vpc"
    run aws ec2 delete-vpc --vpc-id "$vpc" --region "$region"
    ok "$vpc deleted"
  done

  # ── Step 6: DHCP option sets (nuke only — freed after VPC deletion) ────────
  if $NUKE; then
    local dhcp_sets
    dhcp_sets=$(aws ec2 describe-dhcp-options --region "$region" \
      --query "DhcpOptions[].DhcpOptionsId" --output text 2>/dev/null || true)
    for dhcp in $dhcp_sets; do
      info "Deleting DHCP option set: $dhcp"
      run aws ec2 delete-dhcp-options --dhcp-options-id "$dhcp" --region "$region" 2>/dev/null || \
        warn "  Could not delete $dhcp (may still be associated — skipping)"
    done
  fi

  if [[ -z "$vpcs_to_delete" && -z "$REGION_VPC_BILLABLE" ]]; then
    echo "  No VPC resources to delete in $region."
  fi
}

# ── IAM inventory (global — shown once regardless of region count) ────────────

IAM_ROLES=""
OIDC_ARN=""
if $INCLUDE_IAM; then
  section "IAM inventory — global (no region) ══════════════════════"
  IAM_ROLES=$(aws iam list-roles \
    --query "Roles[?starts_with(RoleName,'${PROJECT}')].RoleName" \
    --output text 2>/dev/null || true)
  OIDC_ARN=$(aws iam list-open-id-connect-providers \
    --query "OpenIDConnectProviderList[?contains(Arn,'token.actions.githubusercontent.com')].Arn" \
    --output text 2>/dev/null || true)
  for r in $IAM_ROLES;  do warn "IAM role        : $r  (global)"; done
  [[ -n "$OIDC_ARN" ]]  && warn "OIDC provider   : $OIDC_ARN  (global)"
  [[ -z "$IAM_ROLES" && -z "$OIDC_ARN" ]] && echo "  (no matching IAM resources found)"
fi

# ── Dry-run: show inventory across all regions, then exit ────────────────────

if $DRY_RUN; then
  for r in "${REGIONS_TO_SCAN[@]}"; do
    scan_region "$r"
  done
  echo
  echo "DRY-RUN complete — no resources were deleted."
  exit 0
fi

# ── Full inventory pass (for confirmation prompt) ─────────────────────────────

section "Full inventory across all regions ═══════════════════════"
for r in "${REGIONS_TO_SCAN[@]}"; do
  scan_region "$r"
done
if $INCLUDE_IAM; then
  echo
  warn "IAM roles and OIDC provider are GLOBAL and will also be deleted."
fi

# ── Confirmation ──────────────────────────────────────────────────────────────

echo
if ! $AUTO_YES; then
  echo "Regions targeted: ${REGIONS_TO_SCAN[*]}"
  if $NUKE; then
    read -r -p "NUKE all resources in the above regions? Type 'nuke' to confirm: " confirm
    [[ "$confirm" == "nuke" ]] || { echo "Aborted."; exit 0; }
  else
    read -r -p "Permanently delete all resources listed above? [y/N] " confirm
    [[ "$confirm" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 0; }
  fi
fi

# ── Delete — regional resources ───────────────────────────────────────────────

for r in "${REGIONS_TO_SCAN[@]}"; do
  delete_region "$r"
done

# ── Delete — IAM (global, opt-in) ────────────────────────────────────────────

if $INCLUDE_IAM; then
  section "Deleting IAM — global ════════════════════════════════════"
  for role in $IAM_ROLES; do
    info "Deleting role: $role"
    local_policies=$(aws iam list-role-policies --role-name "$role" \
      --query "PolicyNames[]" --output text 2>/dev/null || true)
    for p in $local_policies; do
      run aws iam delete-role-policy --role-name "$role" --policy-name "$p"
    done
    managed=$(aws iam list-attached-role-policies --role-name "$role" \
      --query "AttachedPolicies[].PolicyArn" --output text 2>/dev/null || true)
    for arn in $managed; do
      run aws iam detach-role-policy --role-name "$role" --policy-arn "$arn"
    done
    run aws iam delete-role --role-name "$role"
    ok "$role deleted"
  done

  if [[ -n "$OIDC_ARN" ]]; then
    info "Deleting OIDC provider: $OIDC_ARN"
    run aws iam delete-open-id-connect-provider --open-id-connect-provider-arn "$OIDC_ARN"
    ok "OIDC provider deleted"
  fi
fi

# ── Summary ───────────────────────────────────────────────────────────────────

section "Done ══════════════════════════════════════════════════════"
echo "  Cleaned regions: ${REGIONS_TO_SCAN[*]}"
if $NUKE; then
  echo
  echo "  NUKE complete. All resources including default VPCs, EC2 instances,"
  echo "  EIPs, DHCP option sets, and Terraform state buckets have been deleted."
else
  if ! $INCLUDE_IAM; then
    echo
    echo "  IAM roles were NOT deleted (pass --include-iam to remove them)."
  fi
  if ! $INCLUDE_VPC; then
    echo
    echo "  VPC resources were REPORTED but not deleted."
    echo "  Pass --include-vpc to remove billable VPC resources and custom VPCs."
    echo "  Pass --nuke to also remove default VPCs and all their resources."
  else
    echo
    echo "  Terraform state bucket(s) were skipped (pass --nuke to include them)."
  fi
fi
