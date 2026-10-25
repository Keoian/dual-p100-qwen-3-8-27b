#pragma once

#include "llama.h"
#include "common.h"

struct common_speculative;

// comma separated list the provided types
std::string common_speculative_type_name_str(const std::vector<enum common_speculative_type> & types);

// comma separated list of all types
const char * common_speculative_all_types_str();

// parse user provided types
std::vector<enum common_speculative_type> common_speculative_types_from_names(const std::vector<std::string> & names);

// infer the spec types from the GGUF metadata of a draft model; empty if unknown
std::vector<enum common_speculative_type> common_speculative_types_from_gguf(const std::string & path);

// convert string to type
enum common_speculative_type common_speculative_type_from_name(const std::string & name);

// convert type to string
std::string common_speculative_type_to_str(enum common_speculative_type type);

// accept rules: "exact", "escape", "lam:<x>", "esc+<x>" (x in (0, 1)), "all". Throws std::invalid_argument
common_spec_accept common_spec_accept_from_str(const std::string & s);
std::string        common_spec_accept_to_str(const common_spec_accept & a);

// return the max number of draft tokens based on the speculative parameters
int32_t common_speculative_n_max(const common_params_speculative * spec);

// return the max number of draft tokens from the initialized implementations
int32_t common_speculative_n_max(const common_speculative * spec);

// validate and resolve the unconditional synthetic acceptance rates
std::vector<double> common_speculative_synth_rates_resolve(const common_params_speculative * spec, int32_t n_max);

// return the conditional synthetic acceptance probabilities
const std::vector<double> & common_speculative_get_synth_probs(const common_speculative * spec);

common_params common_base_params_to_speculative(const common_params & params);

struct common_speculative_output_limits {
    int32_t total;
    int32_t per_seq;
};

// return the output limits needed for speculative decoding
common_speculative_output_limits common_speculative_get_output_limits(
        int32_t n_batch, int32_t n_parallel, int32_t n_draft);

common_speculative * common_speculative_init(common_params_speculative & params, uint32_t n_seq);

void common_speculative_free(common_speculative * spec);

struct common_speculative_draft_params {
    // this flag is used to chain the drafts through all the available implementations
    // after the first successful draft from an implementation, we set it
    //   to false to prevent further drafts for that sequence
    // at the end of the draft() call, all drafting flags will be reset to false
    bool drafting = false;

    // overrides individual configurations (-1 disabled)
    // can be used to constraint the max draft based on the remaining context size
    int32_t n_max = -1;

    llama_pos   pos0;
    llama_token id_last;

    // TODO: remove in the future by keeping track of the prompt from the _begin() call and the consecutive accept calls
    const llama_tokens * prompt;

    // the generated draft from the last _draft() call
    llama_tokens * result;

    // when the draft was sampled rather than taken greedily: for each drafted token, the
    // distribution it was drawn from (candidates with normalized p). the target then verifies
    // with the speculative-sampling rule (common_sampler_sample_and_accept_n_dist). empty for
    // greedy drafts.
    std::vector<std::vector<llama_token_data>> dists = {};

    // the target sampler's temperature for this sequence's request (< 0: unknown). LLAMA_SPEC_STOP_ISO applies only
    // when it lies in LLAMA_SPEC_STOP_ISO_TEMP (or is unknown)
    float target_temp = -1.0f;

    // the request accepts drafts by an approximate rule (common_spec_accept::approx): the MTP draft then samples each
    // token with the request's own settings (below) and drafts without the stop tables or p-min, as measured
    bool accept_approx = false;

    // the rest of the request's sampler settings, for accept_approx (drafts sampled the way the target samples)
    int32_t      target_top_k     = 0;
    float        target_top_p     = 1.0f;
    float        target_min_p     = 0.0f;
    float        target_present   = 0.0f;
    int32_t      target_pen_last_n = 0;
    llama_tokens pen_window       = {}; // the target's penalty window up to and including id_last, oldest first
};

// whether any configured draft type reads common_speculative_draft_params::prompt when drafting (the
// MTP draft does not); a caller can skip rebuilding it every cycle otherwise
bool common_speculative_draft_reads_prompt(const common_speculative * spec);

common_speculative_draft_params & common_speculative_get_draft_params(common_speculative * spec, llama_seq_id seq_id);

// optionally call once at the beginning of a new generation
// reseed the draft's own randomness (sampled drafts), e.g. from the request's sampling seed
void common_speculative_set_seed(common_speculative * spec, uint32_t seed);

void common_speculative_begin(common_speculative * spec, llama_seq_id seq_id, const llama_tokens & prompt);

// process the batch and update the internal state of the speculative context
bool common_speculative_process(common_speculative * spec, const llama_batch & batch);

// generate drafts for the sequences specified with `common_speculative_get_draft_params`
void common_speculative_draft(common_speculative * spec);

// informs the speculative context that n_accepted tokens were accepted by the target model
void common_speculative_accept(common_speculative * spec, llama_seq_id, uint16_t n_accepted);

// (optional) get/set internal state
bool common_speculative_get_state(common_speculative * spec, llama_seq_id seq_id, std::vector<uint8_t> & data);
void common_speculative_set_state(common_speculative * spec, llama_seq_id seq_id, const std::vector<uint8_t> & data);

// print statistics about the speculative decoding
void common_speculative_print_stats(const common_speculative * spec);

struct common_speculative_deleter {
    void operator()(common_speculative * s) { common_speculative_free(s); }
};

typedef std::unique_ptr<common_speculative, common_speculative_deleter> common_speculative_ptr;

struct common_speculative_init_result {
    common_speculative_init_result(common_params & params, llama_model * model_tgt, llama_context * ctx_tgt);
    ~common_speculative_init_result();

    llama_model   * model();
    llama_context * context();

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using common_speculative_init_result_ptr = std::unique_ptr<common_speculative_init_result>;

common_speculative_init_result_ptr common_speculative_init_from_params(common_params & params, llama_model * model_tgt, llama_context * ctx_tgt);
