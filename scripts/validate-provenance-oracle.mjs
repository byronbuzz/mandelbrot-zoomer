const INVALID = 0;
const ESCAPED = 1;
const ANALYTIC = 2;
const PROVISIONAL = 3;
const FINAL = 4;
const CONFLICT = 5;
const ESCAPED_ESTIMATE = 6;
const ANALYTIC_ESTIMATE = 7;

const item = (kind, evidence = 0, rank = 0, origin = 1) => ({ kind, evidence, rank, origin });
const invalid = item(INVALID);

function isDefinite(kind) { return kind === ESCAPED || kind === ANALYTIC; }
function isEstimate(kind) { return kind === ESCAPED_ESTIMATE || kind === ANALYTIC_ESTIMATE; }
function isCap(kind) { return kind === PROVISIONAL || kind === FINAL; }
function classificationFamily(kind) {
  if (kind === ESCAPED || kind === ESCAPED_ESTIMATE) return 1;
  if (kind === ANALYTIC || kind === ANALYTIC_ESTIMATE) return 2;
  return 0;
}

function representative(left, right) {
  if (right.rank !== left.rank) return right.rank > left.rank ? right : left;
  return right.origin > left.origin ? right : left;
}

function merge(left, right) {
  if (left.kind === INVALID) return right;
  if (right.kind === INVALID) return left;
  if (left.kind === CONFLICT) return left;
  if (right.kind === CONFLICT) return right;
  const leftDefinite = isDefinite(left.kind);
  const rightDefinite = isDefinite(right.kind);
  if (leftDefinite && rightDefinite) {
    const classConflict = left.kind !== right.kind;
    const payloadConflict = !classConflict
      && left.kind === ESCAPED
      && left.evidence !== right.evidence;
    if (classConflict || payloadConflict) {
      const display = representative(left, right);
      return { ...display, kind: CONFLICT, evidence: payloadConflict ? 2 : 1 };
    }
  } else {
    if (leftDefinite) return left;
    if (rightDefinite) return right;
  }
  if (isEstimate(left.kind) && isCap(right.kind)) return left;
  if (isEstimate(right.kind) && isCap(left.kind)) return right;
  if (left.kind === FINAL && right.kind === PROVISIONAL) return left;
  if (right.kind === FINAL && left.kind === PROVISIONAL) return right;
  if (isCap(left.kind) && isCap(right.kind) && left.evidence !== right.evidence) {
    return right.evidence > left.evidence ? right : left;
  }
  if (right.rank !== left.rank) return right.rank > left.rank ? right : left;
  if (right.origin !== left.origin) return right.origin > left.origin ? right : left;
  if (right.evidence !== left.evidence) return right.evidence > left.evidence ? right : left;
  return left;
}

function presentMerge(history, current) {
  const historyFamily = classificationFamily(history.kind);
  const preserveFinerCompatibleHistory = history.origin !== 3 && current.origin === 3
    && historyFamily !== 0 && historyFamily === classificationFamily(current.kind)
    && history.rank > current.rank;
  return preserveFinerCompatibleHistory ? history : merge(history, current);
}

function normalize(value, exactResolvedView) {
  if (exactResolvedView) return value;
  if (value.kind === FINAL) return { ...value, kind: PROVISIONAL };
  if (value.kind === ESCAPED) return { ...value, kind: ESCAPED_ESTIMATE };
  if (value.kind === ANALYTIC) return { ...value, kind: ANALYTIC_ESTIMATE };
  return value;
}

function fold(values) { return values.reduce(merge, invalid); }
function semantic(value) { return `${value.kind}:${value.evidence}:${value.rank}`; }
function expect(label, actual, expected) {
  if (semantic(actual) !== semantic(expected)) {
    throw new Error(`${label}: expected ${semantic(expected)}, received ${semantic(actual)}`);
  }
}

const cases = [
  ['invalid accepts cap', [invalid, item(PROVISIONAL, 100, 2)], item(PROVISIONAL, 100, 2)],
  ['escaped survives provisional cap', [item(ESCAPED, 80, 1), item(PROVISIONAL, 5000, 9)], item(ESCAPED, 80, 1)],
  ['analytic survives final cap', [item(ANALYTIC, 1, 1), item(FINAL, 1000, 9)], item(ANALYTIC, 1, 1)],
  ['definite replaces cap', [item(PROVISIONAL, 100, 8), item(ESCAPED, 80, 1)], item(ESCAPED, 80, 1)],
  ['compatible escape uses footprint', [item(ESCAPED, 80, 1), item(ESCAPED, 80, 7)], item(ESCAPED, 80, 7)],
  ['escape payload conflict', [item(ESCAPED, 80, 7), item(ESCAPED, 81, 7, 3)], item(CONFLICT, 2, 7, 3)],
  ['exact escape payload conflicts across footprint', [item(ESCAPED, 80, 7), item(ESCAPED, 81, 9)], item(CONFLICT, 2, 9)],
  ['class conflict', [item(ESCAPED, 80, 7), item(ANALYTIC, 1, 7, 3)], item(CONFLICT, 1, 7, 3)],
  ['exact class conflicts across footprint', [item(ESCAPED, 80, 7), item(ANALYTIC, 1, 9)], item(CONFLICT, 1, 9)],
  ['exact replaces spatial estimate', [item(ESCAPED_ESTIMATE, 80, 9), item(ANALYTIC, 1, 7)], item(ANALYTIC, 1, 7)],
  ['estimate preserves colour over cap', [item(ESCAPED_ESTIMATE, 80, 4), item(PROVISIONAL, 5000, 9)], item(ESCAPED_ESTIMATE, 80, 4)],
  ['compatible analytic uses footprint', [item(ANALYTIC, 1, 2), item(ANALYTIC, 1, 8)], item(ANALYTIC, 1, 8)],
  ['cap frontier precedes footprint', [item(PROVISIONAL, 100, 9), item(PROVISIONAL, 200, 1)], item(PROVISIONAL, 200, 1)],
  ['cap footprint breaks frontier tie', [item(PROVISIONAL, 200, 1), item(PROVISIONAL, 200, 8)], item(PROVISIONAL, 200, 8)],
  ['exact final precedes estimate', [item(FINAL, 1000, 1), item(PROVISIONAL, 5000, 9)], item(FINAL, 1000, 1)],
  ['conflict is sticky against cap', [item(CONFLICT, 1, 4), item(PROVISIONAL, 12000, 9)], item(CONFLICT, 1, 4)]
];

for (const [label, values, expected] of cases) expect(label, fold(values), expected);
expect('reprojected final is provisional', normalize(item(FINAL, 1000, 4), false), item(PROVISIONAL, 1000, 4));
expect('reprojected escape is estimate', normalize(item(ESCAPED, 80, 4), false), item(ESCAPED_ESTIMATE, 80, 4));
expect('exact final remains final', normalize(item(FINAL, 1000, 4), true), item(FINAL, 1000, 4));
expect('coarse current escape preserves finer history',
  presentMerge(item(ESCAPED_ESTIMATE, 80, 9), item(ESCAPED, 80, 7, 3)),
  item(ESCAPED_ESTIMATE, 80, 9));
expect('finer current escape replaces coarser history',
  presentMerge(item(ESCAPED_ESTIMATE, 80, 7), item(ESCAPED, 80, 9, 3)),
  item(ESCAPED, 80, 9, 3));
expect('equal footprint prefers current exact escape',
  presentMerge(item(ESCAPED_ESTIMATE, 80, 9), item(ESCAPED, 80, 9, 3)),
  item(ESCAPED, 80, 9, 3));
expect('coarse incompatible current records conflict with finer history representative',
  presentMerge(item(ESCAPED, 80, 9), item(ANALYTIC, 1, 7, 3)),
  item(CONFLICT, 1, 9));
expect('finer incompatible current records conflict with current representative',
  presentMerge(item(ESCAPED, 80, 7), item(ANALYTIC, 1, 9, 3)),
  item(CONFLICT, 1, 9, 3));

function permutations(values) {
  if (values.length <= 1) return [values];
  return values.flatMap((value, index) => permutations(values.filter((_, other) => other !== index))
    .map(rest => [value, ...rest]));
}

const invariantSets = [
  [item(ESCAPED, 80, 2), item(PROVISIONAL, 5000, 9), item(ESCAPED, 80, 7)],
  [item(ANALYTIC, 1, 2), item(PROVISIONAL, 5000, 9), item(ANALYTIC, 1, 7)],
  [item(PROVISIONAL, 100, 9), item(PROVISIONAL, 200, 1), item(PROVISIONAL, 200, 8)],
  [item(ESCAPED, 80, 7), item(ANALYTIC, 1, 7), item(ESCAPED_ESTIMATE, 80, 9)]
];
for (const values of invariantSets) {
  const expected = semantic(fold(values));
  for (const permutation of permutations(values)) {
    if (semantic(fold(permutation)) !== expected) throw new Error(`Permutation invariant failed: ${expected}`);
  }
  for (const value of values) expect('idempotence', merge(value, value), value);
  if (values.length === 3) {
    const leftAssociated = merge(merge(values[0], values[1]), values[2]);
    const rightAssociated = merge(values[0], merge(values[1], values[2]));
    if (semantic(leftAssociated) !== semantic(rightAssociated)) {
      throw new Error(`Associativity invariant failed: ${semantic(leftAssociated)} != ${semantic(rightAssociated)}`);
    }
  }
}

console.log(`Validated ${cases.length + 8} provenance oracle cases plus permutation, idempotence, and associativity invariants.`);
