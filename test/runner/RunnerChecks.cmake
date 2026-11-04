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
string(REPLACE "pipeline: tensor-cpu" "pipeline: tensor-metal" changed "${valid}")
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
message(STATUS "Runner configuration, assertion and selection contracts passed")
