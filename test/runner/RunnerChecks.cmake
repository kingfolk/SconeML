# Test the runner itself: invalid configurations/assertions must never go green.
if(NOT DEFINED RUNNER OR NOT DEFINED DATA_ROOT OR NOT DEFINED SCRATCH)
  message(FATAL_ERROR "RUNNER, DATA_ROOT and SCRATCH are required")
endif()
file(MAKE_DIRECTORY "${SCRATCH}" "${SCRATCH}/empty")
function(expect_failure label expected)
  execute_process(COMMAND "${RUNNER}" ${ARGN}
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE errors)
  string(FIND "${output}${errors}" "${expected}" found)
  if(result EQUAL 0 OR found EQUAL -1)
    message(FATAL_ERROR "${label}: exit=${result}; expected '${expected}'\n${output}${errors}")
  endif()
endfunction()
function(expect_success label expected)
  execute_process(COMMAND "${RUNNER}" ${ARGN}
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE errors)
  string(FIND "${output}${errors}" "${expected}" found)
  if(NOT result EQUAL 0 OR found EQUAL -1)
    message(FATAL_ERROR "${label}: exit=${result}; expected '${expected}'\n${output}${errors}")
  endif()
endfunction()
set(valid [=[(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: tensor-cpu
*)
(* @assert:cpu:result
[4, 9, 16]
*)
let xs = [1, 2, 3] in xs * xs + 2 * xs + 1
]=])
file(WRITE "${SCRATCH}/valid.ml" "${valid}")
expect_success(valid "1 passed" --file "${SCRATCH}/valid.ml")
expect_failure(missing-file "file not found" --file no-such-case.ml)
expect_failure(missing-run "run not found" --file "${SCRATCH}/valid.ml" --run absent)
expect_failure(empty-suite "empty test suite" --suite "${SCRATCH}/empty")
expect_failure(strict-device "required device unavailable" --no-metal --require-metal
  --file "${DATA_ROOT}/metal_exporter/shared_square.mlir" --run compile)

string(REPLACE "version: 1" "version: 1\nextra: typo" changed "${valid}")
file(WRITE "${SCRATCH}/unknown-key.ml" "${changed}")
expect_failure(unknown-key "unknown key" --file "${SCRATCH}/unknown-key.ml")
string(REPLACE "version: 1" "version: 1\nversion: 1" changed "${valid}")
file(WRITE "${SCRATCH}/duplicate-key.ml" "${changed}")
expect_failure(duplicate-key "duplicated mapping key" --file "${SCRATCH}/duplicate-key.ml")
string(REPLACE "pipeline: tensor-cpu" "pipeline: nonexistent" changed "${valid}")
file(WRITE "${SCRATCH}/unknown-pipeline.ml" "${changed}")
expect_failure(unknown-pipeline "unknown pipeline" --file "${SCRATCH}/unknown-pipeline.ml")
string(REPLACE "pipeline: tensor-cpu" "pipeline: tensor-cpu\n  - name: cpu\n    pipeline: tensor-cpu" changed "${valid}")
file(WRITE "${SCRATCH}/duplicate-run.ml" "${changed}")
expect_failure(duplicate-run "duplicate run" --file "${SCRATCH}/duplicate-run.ml")
string(REPLACE "@assert:cpu:result" "@assert:cpu:typo" changed "${valid}")
file(WRITE "${SCRATCH}/unknown-assert.ml" "${changed}")
expect_failure(unknown-assert "unknown assertion" --file "${SCRATCH}/unknown-assert.ml")
string(REPLACE "@assert:cpu:result" "@assert:orphan:result" changed "${valid}")
file(WRITE "${SCRATCH}/missing-assert.ml" "${changed}")
expect_failure(missing-assert "no assertions" --file "${SCRATCH}/missing-assert.ml")
string(REPLACE "let xs =" "(* @assert:orphan:result\n[4]\n*)\nlet xs =" changed "${valid}")
file(WRITE "${SCRATCH}/orphan-assert.ml" "${changed}")
expect_failure(orphan-assert "unknown run" --file "${SCRATCH}/orphan-assert.ml")
string(REPLACE "let xs =" "(* @assert:cpu:result\n[4]\n*)\nlet xs =" changed "${valid}")
file(WRITE "${SCRATCH}/duplicate-assert.ml" "${changed}")
expect_failure(duplicate-assert "duplicate assertion" --file "${SCRATCH}/duplicate-assert.ml")
string(REPLACE "[4, 9, 16]" "[5, 9, 16]" changed "${valid}")
file(WRITE "${SCRATCH}/wrong-result.ml" "${changed}")
expect_failure(wrong-result "expected [5, 9, 16]" --file "${SCRATCH}/wrong-result.ml")
string(REPLACE "[4, 9, 16]" "4" changed "${valid}")
file(WRITE "${SCRATCH}/wrong-result-type.ml" "${changed}")
expect_failure(wrong-result-type "expected 4" --file "${SCRATCH}/wrong-result-type.ml")
string(REPLACE "@assert:cpu:result" "@assert:cpu:effects" changed "${valid}")
file(WRITE "${SCRATCH}/wrong-assert-target.ml" "${changed}")
expect_failure(wrong-assert-target "not supported by" --file "${SCRATCH}/wrong-assert-target.ml")
string(REPLACE "let xs = [1, 2, 3] in xs * xs + 2 * xs + 1" "" changed "${valid}")
file(WRITE "${SCRATCH}/no-program.ml" "${changed}")
expect_failure(no-program "missing test program" --file "${SCRATCH}/no-program.ml")
string(REPLACE "@assert:cpu:result\n[4, 9, 16]" "@assert:cpu:compare\nabsent" changed "${valid}")
file(WRITE "${SCRATCH}/missing-compare.ml" "${changed}")
expect_failure(missing-compare "compare must reference" --file "${SCRATCH}/missing-compare.ml")
string(REPLACE "pipeline: tensor-cpu" "pipeline: tensor-msl" changed "${valid}")
string(REPLACE "@assert:cpu:result\n[4, 9, 16]" "@assert:cpu:text-check\nCHCEK: typo" changed "${changed}")
file(WRITE "${SCRATCH}/bad-filecheck.ml" "${changed}")
expect_failure(bad-filecheck "invalid FileCheck assertion" --file "${SCRATCH}/bad-filecheck.ml")
file(WRITE "${SCRATCH}/wrong-stage.mlir" [=[// @test
// version: 1
// input: mlir
// runs:
//   - name: bad
//     pipeline: tensor-msl
// @end
// @assert:bad:error
// stage: verify
// message: exactly one gpu.module
// @end
module {}
]=])
expect_failure(wrong-stage "Actual stage: msl-export" --file "${SCRATCH}/wrong-stage.mlir")
file(WRITE "${SCRATCH}/unexpected-success.mlir" [=[// @test
// version: 1
// input: mlir
// runs:
//   - name: bad
//     pipeline: verify
// @end
// @assert:bad:error
// stage: verify
// message: rejected
// @end
module {}
]=])
expect_failure(unexpected-success "expected error at verify" --file "${SCRATCH}/unexpected-success.mlir")
expect_success(legacy "passed" --file "${DATA_ROOT}/let.ml")
expect_success(selected-compare "[cpu]" --file "${DATA_ROOT}/tensor/polynomial.ml" --run metal)
expect_success(skip-compare "1 skipped" --file "${DATA_ROOT}/tensor/polynomial.ml" --run metal --no-metal)

string(REPLACE "pipeline: tensor-cpu" "pipeline: tensor-cpu\n    threads-per-group: 4" changed "${valid}")
file(WRITE "${SCRATCH}/wrong-launch-pipeline.ml" "${changed}")
expect_failure(wrong-launch-pipeline "only supported by tensor-metal" --file "${SCRATCH}/wrong-launch-pipeline.ml")
string(REPLACE "pipeline: tensor-cpu" "pipeline: tensor-metal\n    threads-per-group: 0" changed "${valid}")
file(WRITE "${SCRATCH}/zero-width.ml" "${changed}")
expect_failure(zero-width "must be positive" --file "${SCRATCH}/zero-width.ml")
string(REPLACE "pipeline: tensor-cpu" "pipeline: tensor-metal\n    threads-per-group: -1" changed "${valid}")
file(WRITE "${SCRATCH}/negative-width.ml" "${changed}")
expect_failure(negative-width "invalid YAML" --file "${SCRATCH}/negative-width.ml")
expect_failure(strict-execute-device "required device unavailable" --no-metal --require-metal
  --file "${DATA_ROOT}/tensor/polynomial.ml" --run metal)

# Raw IR execution has explicit host data and a deliberately small wrapper ABI.
file(READ "${DATA_ROOT}/acceptance/ir/001-add.mlir" map_case)
string(REPLACE "// tensor-input: [2147483647]\n" "" changed "${map_case}")
file(WRITE "${SCRATCH}/missing-input.mlir" "${changed}")
expect_failure(missing-input "requires mlir input and tensor-input" --file "${SCRATCH}/missing-input.mlir")
string(REPLACE "// tensor-input: [2147483647]" "// tensor-input: []" changed "${map_case}")
file(WRITE "${SCRATCH}/wrong-input-length.mlir" "${changed}")
expect_failure(wrong-input-length "length must match" --file "${SCRATCH}/wrong-input-length.mlir" --run cpu)
string(REPLACE "// tensor-input: [2147483647]" "// tensor-input: [2147483648]" changed "${map_case}")
file(WRITE "${SCRATCH}/wide-input.mlir" "${changed}")
expect_failure(wide-input "outside i32 range" --file "${SCRATCH}/wide-input.mlir")
string(REPLACE "memref<1xi32>" "memref<?xi32>" changed "${map_case}")
file(WRITE "${SCRATCH}/dynamic-wrapper.mlir" "${changed}")
expect_failure(dynamic-wrapper "static identity-layout host buffer arguments" --file "${SCRATCH}/dynamic-wrapper.mlir" --run cpu)
string(REPLACE "// @assert:cpu:result\n// [-2147483648]" "// @assert:cpu:result\n// [0]" changed "${map_case}")
file(WRITE "${SCRATCH}/wrong-ir-result.mlir" "${changed}")
expect_failure(wrong-ir-result "first differing element [0]" --file "${SCRATCH}/wrong-ir-result.mlir" --run cpu)
string(REPLACE "input: source" "input: source\ntensor-input: [1]" changed "${valid}")
file(WRITE "${SCRATCH}/source-host-input.ml" "${changed}")
expect_failure(source-host-input "tensor-input requires mlir input" --file "${SCRATCH}/source-host-input.ml")
expect_success(selected-ir-compare "[cpu]" --file "${DATA_ROOT}/acceptance/ir/001-add.mlir" --run metal --no-metal)
expect_failure(strict-ir-device "required device unavailable" --file "${DATA_ROOT}/acceptance/ir/001-add.mlir" --run metal --no-metal --require-metal)
string(REPLACE "    func.return" "    %noise = arith.constant 1 : i32\n    func.return" changed "${map_case}")
file(WRITE "${SCRATCH}/extra-wrapper-op.mlir" "${changed}")
expect_failure(extra-wrapper-op "only a tensor_map followed by func.return" --file "${SCRATCH}/extra-wrapper-op.mlir" --run cpu)
file(WRITE "${SCRATCH}/unused-host-input.mlir" [=[// @test
// version: 1
// input: mlir
// tensor-input: [1]
// runs:
//   - name: check
//     pipeline: verify
// @end
// @assert:check:success
// true
// @end
module {}
]=])
expect_failure(unused-host-input "tensor-input requires an IR execution run" --file "${SCRATCH}/unused-host-input.mlir")

# Production external-input source runs have separate per-run named arguments.
file(READ "${DATA_ROOT}/external_input/identity.ml" source_inputs)
string(REPLACE "    arguments:" "    typo-arguments:" changed "${source_inputs}")
file(WRITE "${SCRATCH}/unknown-arguments.ml" "${changed}")
expect_failure(unknown-arguments "unknown key" --file "${SCRATCH}/unknown-arguments.ml")
string(REPLACE "pipeline: tensor-input-cpu" "pipeline: tensor-cpu" changed "${source_inputs}")
file(WRITE "${SCRATCH}/literal-arguments.ml" "${changed}")
expect_failure(literal-arguments "arguments is only supported" --file "${SCRATCH}/literal-arguments.ml")
string(REPLACE "values:" "extra: 0\n        values:" changed "${source_inputs}")
file(WRITE "${SCRATCH}/unknown-argument-field.ml" "${changed}")
expect_failure(unknown-argument-field "unknown key" --file "${SCRATCH}/unknown-argument-field.ml")
string(REPLACE "2147483647" "2147483648" changed "${source_inputs}")
file(WRITE "${SCRATCH}/wide-source-value.ml" "${changed}")
expect_failure(wide-source-value "invalid YAML" --file "${SCRATCH}/wide-source-value.ml")
string(REPLACE "pipeline: tensor-input-cpu" "pipeline: tensor-input-unknown" changed "${source_inputs}")
file(WRITE "${SCRATCH}/unimplemented-source-metal.ml" "${changed}")
expect_failure(unimplemented-source-metal "unknown pipeline" --file "${SCRATCH}/unimplemented-source-metal.ml" --no-metal)
file(WRITE "${SCRATCH}/missing-arguments.ml" [=[(* @test
version: 1
input: source
runs:
  - name: cpu
    pipeline: tensor-input-cpu
*)
(* @assert:cpu:result
[1]
*)
xs
]=])
expect_failure(missing-arguments "requires source input and arguments" --file "${SCRATCH}/missing-arguments.ml")

# Device policy for the production external-input GPU API, including no-op and
# errors that must be checked before any device availability decision.
expect_success(external-metal-skip "1 skipped" --file "${DATA_ROOT}/external_input/polynomial.ml" --run gpu-first --no-metal)
expect_failure(external-metal-required "required device unavailable" --file "${DATA_ROOT}/external_input/polynomial.ml" --run gpu-first --no-metal --require-metal)
expect_success(external-metal-empty "2 passed, 0 skipped" --file "${DATA_ROOT}/external_input/identity.ml" --run gpu-empty --no-metal --require-metal)
expect_success(external-metal-invalid-expression "1 passed, 0 skipped" --file "${DATA_ROOT}/external_input/errors/empty-unknown.ml" --run gpu-rejected --no-metal --require-metal)
expect_success(external-metal-invalid-width "2 passed, 0 skipped" --file "${DATA_ROOT}/external_input/metal/invalid-width.ml" --no-metal --require-metal)
message(STATUS "Runner configuration, assertion, launch and source/IR input device-policy contracts passed")

# Multi-input source calls remain data-driven and retain strict device policy.
expect_success(multi-cpu "1 passed" --file "${DATA_ROOT}/multi_input/three-fma.ml" --run cpu)
expect_success(multi-metal-skip "1 skipped" --file "${DATA_ROOT}/multi_input/three-fma.ml" --run metal --no-metal)
expect_failure(multi-metal-required "required device unavailable" --file "${DATA_ROOT}/multi_input/three-fma.ml" --run metal --no-metal --require-metal)
expect_success(multi-empty "2 passed, 0 skipped" --file "${DATA_ROOT}/multi_input/three-empty.ml" --no-metal --require-metal)
expect_success(multi-shape-before-skip "2 passed, 0 skipped" --file "${DATA_ROOT}/multi_input/errors/empty-mismatched.ml" --no-metal --require-metal)
