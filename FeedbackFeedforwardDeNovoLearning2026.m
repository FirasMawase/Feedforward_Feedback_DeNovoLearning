% ========================================================================
% Feedforward-Feedback dynamic shapes de novo motor learning
%
% Simulates 2D tracking during baseline, mirror-reversal learning, and washout.
% The model combines a cortical RNN feedforward controller with a
% cerebellar-like feedback controller driven by delayed sensory prediction errors.
%
% Authors: Firas Mawase & Chen Avraham, March 2026
% ========================================================================

clearvars; clc; close all;

%% ------------------------------------------------------------------------
% SIMULATION SETTINGS
% ------------------------------------------------------------------------
nSubjects    = 10;  % use 2 for a quick check
nTrials_Pret = 720; % pretraining
nTrials_Base = 100; % baseline
nTrials_MR   = 360; % mirror-reversal learning
nTrials_Wash = 30;  % washout
nFreq = 7;

cond_names = {'LocalFeedback'};  % local feedback-tutor recurrent update
                                 % This local Learning rule is based on the work "A neural implementation model of feedback-based motor learning"
                                 % by Barbara Feulner et al., Nature Communications 16:1805 (2025)
nConds = numel(cond_names);

%% Initialization 

% Data storage (cond x subj x trial x freq)
pop_OG_total      = zeros(nConds, nSubjects, nTrials_MR, nFreq);
pop_OG_FF_open    = zeros(nConds, nSubjects, nTrials_MR, nFreq);
pop_OG_FB_open    = zeros(nConds, nSubjects, nTrials_MR, nFreq);
pop_OG_FF_closed  = zeros(nConds, nSubjects, nTrials_MR, nFreq);
pop_OG_FB_closed  = zeros(nConds, nSubjects, nTrials_MR, nFreq);

pop_RMS_FF_full   = zeros(nConds, nSubjects, nTrials_MR);
pop_RMS_FB_full   = zeros(nConds, nSubjects, nTrials_MR);

pop_MSE_full      = zeros(nConds, nSubjects, nTrials_MR);
pop_MSE_FF_open   = zeros(nConds, nSubjects, nTrials_MR);
pop_MSE_FB_open   = zeros(nConds, nSubjects, nTrials_MR);
pop_MSE_FF_closed = zeros(nConds, nSubjects, nTrials_MR);
pop_MSE_FB_closed = zeros(nConds, nSubjects, nTrials_MR);

pop_OG_total_wash      = zeros(nConds, nSubjects, nFreq);
pop_OG_FF_closed_wash  = zeros(nConds, nSubjects, nFreq);
pop_OG_FB_closed_wash  = zeros(nConds, nSubjects, nFreq);

% Store washout OG per trial (for plotting washout decay)
pop_OG_total_wash_trials = zeros(nConds, nSubjects, nTrials_Wash, nFreq);


% Diagnostics
pop_SPErms_vis  = nan(nConds, nSubjects, nTrials_MR);
pop_predR2_vis  = nan(nConds, nSubjects, nTrials_MR);
pop_dWrec_rel   = zeros(nConds, nSubjects, nTrials_MR);

pop_drive_rec_rms = zeros(nConds, nSubjects, nTrials_MR);
pop_drive_inp_rms = zeros(nConds, nSubjects, nTrials_MR);
pop_drive_fb_rms  = zeros(nConds, nSubjects, nTrials_MR);

% Baseline tail MSE buffer (for error plot baseline then MR)
Wb_mean = 20;   % baseline mean window for AE
Wb_plot = 50;   % number of baseline trials to show before MR
Wb = Wb_plot;   % buffer length stored in pop_OG_*_base_buf
pop_MSE_base_buf = nan(nConds, nSubjects, Wb_plot);

% Wrec diagnostics (drift, alignment, low-rank snapshots)
pop_Wrec_drift = zeros(nConds, nSubjects, nTrials_MR);   % ||Wrec(tr)-Wrec(MR start)|| / ||Wrec(MR start)||
pop_Wrec_align = nan(nConds, nSubjects, nTrials_MR);     % cosine alignment between actual Delta Wrec and local update
snap_stride = 10;                                        % save Wrec snapshot every 10 MR trials
nSnaps = floor(nTrials_MR / snap_stride);
Wrec_snaps = cell(nConds, nSubjects);                    % each cell: Nff x Nff x nSnaps
Wrec_ref_store = cell(nConds, nSubjects);                % Wrec reference at MR start (per cond/subj)

% Teaching-direction snapshots for mode projections (dW ~ post * eligibility')
post_snaps = cell(nConds, nSubjects);   % each cell: Nff x nSnaps (post direction)
elig_snaps = cell(nConds, nSubjects);   % each cell: Nff x nSnaps (eligibility direction)


% Baseline mean OG buffers and means (for after-effects)
pop_OG_total_base_buf     = zeros(nConds, nSubjects, Wb, nFreq);
pop_OG_FF_closed_base_buf = zeros(nConds, nSubjects, Wb, nFreq);
pop_OG_FB_closed_base_buf = zeros(nConds, nSubjects, Wb, nFreq);

pop_OG_total_base_mean     = zeros(nConds, nSubjects, nFreq);
pop_OG_FF_closed_base_mean = zeros(nConds, nSubjects, nFreq);
pop_OG_FB_closed_base_mean = zeros(nConds, nSubjects, nFreq);

aeT = squeeze(pop_OG_total_wash(nConds,nSubjects,:))     - squeeze(pop_OG_total_base_mean(nConds,nSubjects,:));
aeF = squeeze(pop_OG_FF_closed_wash(nConds,nSubjects,:)) - squeeze(pop_OG_FF_closed_base_mean(nConds,nSubjects,:));
aeB = squeeze(pop_OG_FB_closed_wash(nConds,nSubjects,:)) - squeeze(pop_OG_FB_closed_base_mean(nConds,nSubjects,:));

pop_AE_total(nConds,nSubjects,:)     = reshape(aeT, 1, 1, []);
pop_AE_FF_closed(nConds,nSubjects,:) = reshape(aeF, 1, 1, []);
pop_AE_FB_closed(nConds,nSubjects,:) = reshape(aeB, 1, 1, []);

% Store trajectories for early vs late MR visualization
traj_early = cell(nConds, nSubjects);
traj_late  = cell(nConds, nSubjects);

%% ------------------------------------------------------------------------
% GLOBAL TASK / PLANT (same for everyone)
% ------------------------------------------------------------------------
dt      = 0.01;
freqs_x = [0.1 0.25 0.55 0.85 1.15 1.55 2.05];
freqs_y = [0.15 0.35 0.65 0.95 1.45 1.85 2.15];
amp_x   = (0.4/2.31)*[2.31 2.31 2.31 1.76 1.30 0.97 0.73];
amp_y   = (0.4/2.31)*[2.31 2.31 2.31 1.58 1.03 0.81 0.70];
t_vec   = (0:round(40/dt)-1)*dt;

% Plant dynamics
[A_d, B_d, C_hand] = build_arm_dynamics(dt, 2*pi*2.0, 1.0, 120.0);

% Mirror reversal about 45 degrees
theta = pi/4;
v = [cos(theta); sin(theta)];
M_MR = 2*(v*v.') - eye(2);          % reflection about axis v

%% ------------------------------------------------------------------------
% CONFIGURATION
% ------------------------------------------------------------------------
cfg = struct();
cfg.dt = dt;

% Visual delay 70 ms, proprio delay 30 ms
cfg.vis_delay_sec     = 0.070;
cfg.proprio_delay_sec = 0.030;
cfg.vis_delay_steps     = max(0, round(cfg.vis_delay_sec / cfg.dt));      % 7
cfg.proprio_delay_steps = max(0, round(cfg.proprio_delay_sec / cfg.dt));  % 3

% Feedback internal delay-channel: use visual delay for visual channels
cfg.tau_steps_vis  = cfg.vis_delay_steps;

% Noise
cfg.motor_noise_std    = 0.10;
cfg.proprio_noise_std  = 0.20;
cfg.vis_meas_noise_std = 0.00;

% Learning rates
cfg.lr_base      = 0.008;   % FF and BPTT Wrec
cfg.lr_F_adapt   = 0.040;   % feedback NN
cfg.lr_RNN_local = 0.008;   % local Wrec rules
cfg.lr_cb_pred   = 3e-3;    % CB W_state learning

% Regularization
cfg.lambda_u_ff  = 5e-3;
cfg.lambda_u_fb  = 3e-4;
cfg.lambda_pos   = 1e-5;
cfg.lambda_du_ff = 4e-4; 

% RNN dynamics
cfg.rnn_tau_sec = 0.10;
cfg.alpha_rnn   = cfg.dt / cfg.rnn_tau_sec;
cfg.elig_tau_sec = 0.05;  % 50 ms, biologically plausible order


% Feedback error low-pass
cfg.err_lp_tau_sec = 0.10;                 
cfg.alpha_lp       = min(1.0, cfg.dt / cfg.err_lp_tau_sec);

% Feedback gain
cfg.betaF = 3.75;

% Estimator fixed correction gains (fast observer-like)
cfg.K_vis  = 0.20;
cfg.K_prop = 0.35;

% Safety
cfg.u_clip      = 100.0;
cfg.h_clip      = 20.0;
cfg.hF_clip     = 20.0;

cfg.use_state_clip = true;
cfg.x_clip = 2.0;
cfg.v_clip = 30.0;

% Spectral radius control
cfg.rho_power_iters = 25;
cfg.target_rho      = 0.98;

% Grad clipping
cfg.clip_grad = 0.5;

% No transient cursor perturbations are used in this version.

% Cerebellum predictor (GC features, learned W_state)
cfg.use_cb_predictor = true;
cfg.cb_Khist         = cfg.vis_delay_steps;
cfg.cb_Ngc           = 300;
cfg.spe_clip         = 2.0;
cfg.cb_ds_clip       = 0.25;

cfg.tutor_lead_steps = cfg.vis_delay_steps;   % lead by visual delay (default)

% Local learning rule options
cfg.local_stride    = 2;
cfg.local_use_delay = true;
cfg.local_only_MR   = true;

% Gains for feedback NN input channels
cfg.gain_vis_tau  = 0.25;   %
cfg.gain_prop_tau = 0.25;   %

cfg.gain_vis_lp   = 1.4;   %
cfg.gain_prop_lp  = 1.6;   %

cfg.gain_vis_de   = 0.035;  %
cfg.gain_prop_de  = 0.035;

% Tutor-based FF learning during MR
% During MR, ff_track_to_ff is set to 0 so tracking gradients do not train FF directly.
% Instead FF imitates a "tutor" derived from the feedback command.
cfg.tutor_tau_sec  = 0.25;     % tutor low-pass time constant (sec), makes tutoring naturally LF-biased
cfg.alpha_tutor    = min(1.0, cfg.dt / cfg.tutor_tau_sec);
cfg.use_valid_sample_mask = true;
% -------------------------------------------------

cfg.ff_track_to_ff = 1.0;   % default: allow tracking grads to FF
cfg.lambda_tutor   = 0.0;   % default: no tutor loss

% Feedforward policy inputs: x*, v*, context
in_dim_ff = 2 + 2 + 1;

% Feedback controller input dimension
D_in_F = 2*6 + 1;

use_ff = true;

fprintf('========================================================\n');
fprintf('Conditions: %s\n', strjoin(cond_names, ', '));
fprintf('nSubjects=%d\n', nSubjects);
fprintf('========================================================\n');

%% ------------------------------------------------------------------------
% TARGET SEEDS (fixed across all subjects and conditions)
% ------------------------------------------------------------------------
masterSeedTargets = 123;
rng(masterSeedTargets);

trialSeeds_pret_all       = randi(1e9, nTrials_Pret, 1);
trialSeeds_base_all       = randi(1e9, nTrials_Base, 1);
trialSeeds_mr_all         = randi(1e9, nTrials_MR,   1);
trialSeeds_wash_all       = randi(1e9, nTrials_Wash, 1);

trialSeeds_base_noise_all = randi(1e9, nTrials_Base, 1);

%% ------------------------------------------------------------------------
% MAIN SIMULATION LOOP
% ------------------------------------------------------------------------
for subj = 1:nSubjects
    fprintf('\nSubject %d / %d\n', subj, nSubjects);

    rng(subj);

    params_init = init_params_ff_F(128, in_dim_ff, 50, 4, D_in_F, cfg);
    params_init.W_rec_ff = enforce_spectral_radius_det(params_init.W_rec_ff, cfg.target_rho, cfg.rho_power_iters);

    %% --------------------------------------------------------------------
    % CONDITION LOOP
    % --------------------------------------------------------------------
    for c = 1:nConds
        cond = cond_names{c};
        fprintf('\n  Condition: %s\n', cond);

        params = params_init;

        %% 1) PRETRAINING
        for tr = 1:nTrials_Pret
            rng(trialSeeds_pret_all(tr));
            target_traj = make_sum_of_sines_2D_xy(t_vec, freqs_x, freqs_y, amp_x, amp_y);

            noiseSeed = make_subj_noise_seed(trialSeeds_pret_all(tr), subj, 1);
            rng(noiseSeed);
            rng_state_before_forward = rng;

            context = 0;
            M_vis   = eye(2);

            cfg_pre = cfg;
            rng(rng_state_before_forward);
            [~, cache] = forward_trial(target_traj, context, params, A_d, B_d, C_hand, M_vis, ...
                true, cfg.betaF, use_ff, cfg_pre);

            grads = backward_trial(target_traj, context, params, cache, A_d, B_d, C_hand, M_vis, cfg_pre, cfg.betaF);
            grads = clip_gradients(grads, cfg_pre.clip_grad);

            params = apply_updates(params, grads, cache, cfg_pre, 'BPTT');

            if mod(tr, 100) == 0
                fprintf('    Pretraining: %d / %d\n', tr, nTrials_Pret);
            end
        end

        %% 2) BASELINE
        for tr = 1:nTrials_Base
            rng(trialSeeds_base_all(tr));
            target_traj = make_sum_of_sines_2D_xy(t_vec, freqs_x, freqs_y, amp_x, amp_y);

            noiseSeed = make_subj_noise_seed(trialSeeds_base_noise_all(tr), subj, 2);
            rng(noiseSeed);
            rng_state_before_forward = rng;

            context = 0;
            M_vis   = eye(2);

            cfg_base = cfg;
            rng(rng_state_before_forward);
            [~, cache] = forward_trial(target_traj, context, params, A_d, B_d, C_hand, M_vis, ...
                true, cfg.betaF, use_ff, cfg_base);

            % Store baseline OG on last Wb baseline trials (before learning update) ---
            if tr > (nTrials_Base - Wb_plot)
                ib = tr - (nTrials_Base - Wb_plot);


                % Baseline tail tracking error, using the displayed cursor
                pop_MSE_base_buf(c, subj, ib) = mean(sum((target_traj - cache.cursor_disp_traj).^2, 1));
                % Closed-loop probes for baseline mean, matched to washout
                cfg_probeB = cfg_base;
                % Total baseline probe, same noise
                rng(rng_state_before_forward);
                [~, cache_base_probe] = forward_trial(target_traj, context, params, A_d, B_d, C_hand, M_vis, ...
                    true, cfg.betaF, use_ff, cfg_probeB);

                og_base_total = compute_orthogonal_gain_multifit(cache_base_probe.hand_traj, target_traj, t_vec, freqs_x, freqs_y, 2, 'real');

                % FF-only closed-loop baseline probe
                rng(rng_state_before_forward);
                [~, cache_FF_closed_B] = forward_trial(target_traj, context, params, A_d, B_d, C_hand, M_vis, ...
                    false, cfg.betaF, true, cfg_probeB);

                % FB-only closed-loop baseline probe
                rng(rng_state_before_forward);
                [~, cache_FB_closed_B] = forward_trial(target_traj, context, params, A_d, B_d, C_hand, M_vis, ...
                    true, cfg.betaF, false, cfg_probeB);

                og_base_ffC = compute_orthogonal_gain_multifit(cache_FF_closed_B.hand_traj, target_traj, t_vec, freqs_x, freqs_y, 2, 'real');
                og_base_fbC = compute_orthogonal_gain_multifit(cache_FB_closed_B.hand_traj, target_traj, t_vec, freqs_x, freqs_y, 2, 'real');

                og_base_total = max(min(og_base_total, 1), -1);
                og_base_ffC   = max(min(og_base_ffC,   1), -1);
                og_base_fbC   = max(min(og_base_fbC,   1), -1);

                pop_OG_total_base_buf(c, subj, ib, :)     = og_base_total;
                pop_OG_FF_closed_base_buf(c, subj, ib, :) = og_base_ffC;
                pop_OG_FB_closed_base_buf(c, subj, ib, :) = og_base_fbC;
            end
            % -------------------------------------------------------------------------------


            grads = backward_trial(target_traj, context, params, cache, A_d, B_d, C_hand, M_vis, cfg_base, cfg.betaF);
            grads = clip_gradients(grads, cfg_base.clip_grad);

            params = apply_updates(params, grads, cache, cfg_base, 'BPTT');

            if mod(tr, 100) == 0
                fprintf('    Baseline: %d / %d\n', tr, nTrials_Base);
            end
        end

        idxMean = (Wb_plot - Wb_mean + 1) : Wb_plot;

        pop_OG_total_base_mean(c, subj, :)     = squeeze(mean(pop_OG_total_base_buf(c, subj, idxMean, :), 3));
        pop_OG_FF_closed_base_mean(c, subj, :) = squeeze(mean(pop_OG_FF_closed_base_buf(c, subj, idxMean, :), 3));
        pop_OG_FB_closed_base_mean(c, subj, :) = squeeze(mean(pop_OG_FB_closed_base_buf(c, subj, idxMean, :), 3));

        %% 3) MIRROR-REVERSAL LEARNING

        % Random trial picks for trajectory visualization (early and late MR)
        early_window = min(50, nTrials_MR);
        late_window  = min(50, nTrials_MR);
        early_pick_tr = randi(early_window);
        late_pick_tr  = nTrials_MR - late_window + randi(late_window);

        % Wrec reference for drift + low-rank snapshots
        Wrec_ref_mr = params.W_rec_ff;
        Wrec_ref_store{c,subj} = Wrec_ref_mr;
        Nff = size(params.W_rec_ff,1);
        if isempty(Wrec_snaps{c,subj})
            Wrec_snaps{c,subj} = zeros(Nff, Nff, nSnaps);
        end
        for tr = 1:nTrials_MR
            rng(trialSeeds_mr_all(tr));
            target_traj = make_sum_of_sines_2D_xy(t_vec, freqs_x, freqs_y, amp_x, amp_y);

            noiseSeed = make_subj_noise_seed(trialSeeds_mr_all(tr), subj, 3);
            rng(noiseSeed);
            rng_state_before_forward = rng;

            context = 1;
            M_vis   = M_MR;

            cfg_mr = cfg;
            % During adaptation, block direct tracking gradients to FF and train FF by tutor imitation
            cfg_mr.ff_track_to_ff = 0.00;   % avoids direct FF tracking updates during adaptation
            cfg_mr.lambda_tutor   = 0.12;   % tutor strength

            % Full run
            rng(rng_state_before_forward);
            [~, cache_full] = forward_trial(target_traj, context, params, A_d, B_d, C_hand, M_vis, ...
                true, cfg.betaF, use_ff, cfg_mr);

            og_total = compute_orthogonal_gain_multifit(cache_full.hand_traj, target_traj, t_vec, freqs_x, freqs_y, 2, 'real');

            if tr == early_pick_tr
                traj_early{c,subj} = pack_traj_for_plot(target_traj, cache_full.cursor_disp_traj);
            end
            if tr == late_pick_tr
                traj_late{c,subj}  = pack_traj_for_plot(target_traj, cache_full.cursor_disp_traj);
            end


            % Open-loop ablation (from stored commands)
            u_FF_only = cache_full.u_ff_hist;
            u_FB_only = cfg.betaF * cache_full.u_fb_hist;

            traj_FF_open_hand = simulate_trajectory(u_FF_only, A_d, B_d, C_hand, cfg_mr);
            traj_FB_open_hand = simulate_trajectory(u_FB_only, A_d, B_d, C_hand, cfg_mr);

            og_ff_open = compute_orthogonal_gain_multifit(traj_FF_open_hand, target_traj, t_vec, freqs_x, freqs_y, 2, 'real');
            og_fb_open = compute_orthogonal_gain_multifit(traj_FB_open_hand, target_traj, t_vec, freqs_x, freqs_y, 2, 'real');

            % Closed-loop ablation probes
            cfg_probe = cfg_mr;
            cfg_probe.motor_noise_std = cfg.motor_noise_std;

            % FF-only closed-loop probe
            rng(rng_state_before_forward);
            [~, cache_FF_closed] = forward_trial(target_traj, context, params, A_d, B_d, C_hand, M_vis, ...
                false, cfg.betaF, true, cfg_probe);

            % FB-only closed-loop probe
            rng(rng_state_before_forward);
            [~, cache_FB_closed] = forward_trial(target_traj, context, params, A_d, B_d, C_hand, M_vis, ...
                true, cfg.betaF, false, cfg_probe);

            og_ff_closed = compute_orthogonal_gain_multifit(cache_FF_closed.hand_traj, target_traj, t_vec, freqs_x, freqs_y, 2, 'real');
            og_fb_closed = compute_orthogonal_gain_multifit(cache_FB_closed.hand_traj, target_traj, t_vec, freqs_x, freqs_y, 2, 'real');

            % Store OG
            pop_OG_total(c, subj, tr, :)     = og_total;
            pop_OG_FF_open(c, subj, tr, :)   = og_ff_open;
            pop_OG_FB_open(c, subj, tr, :)   = og_fb_open;
            pop_OG_FF_closed(c, subj, tr, :) = og_ff_closed;
            pop_OG_FB_closed(c, subj, tr, :) = og_fb_closed;

            % Effort
            pop_RMS_FF_full(c, subj, tr) = rms(u_FF_only(:));
            pop_RMS_FB_full(c, subj, tr) = rms(u_FB_only(:));

            % Tracking MSE, using the displayed cursor as in the training loss
            pop_MSE_full(c, subj, tr) = tracking_mse_delayed(target_traj, cache_full.cursor_disp_traj, cache_full.valid_sample_mask, cfg.vis_delay_steps);
            traj_FF_open_cursor = M_vis * traj_FF_open_hand;
            traj_FB_open_cursor = M_vis * traj_FB_open_hand;
            pop_MSE_FF_open(c, subj, tr)   = mean(sum((target_traj - traj_FF_open_cursor).^2, 1));
            pop_MSE_FB_open(c, subj, tr)   = mean(sum((target_traj - traj_FB_open_cursor).^2, 1));
            pop_MSE_FF_closed(c, subj, tr) = tracking_mse_delayed(target_traj, cache_FF_closed.cursor_disp_traj, cache_FF_closed.valid_sample_mask, cfg.vis_delay_steps);
            pop_MSE_FB_closed(c, subj, tr) = tracking_mse_delayed(target_traj, cache_FB_closed.cursor_disp_traj, cache_FB_closed.valid_sample_mask, cfg.vis_delay_steps);
            
            % CB diagnostics (visual)
            if isfield(cache_full,'spe_vis_traj') && isfield(cache_full,'x_hat_vis_delayed_traj')
                spev = cache_full.spe_vis_traj;
                pop_SPErms_vis(c,subj,tr) = sqrt(mean(spev(:).^2));

                xdisp = cache_full.cursor_disp_traj;
                xpred = cache_full.x_hat_vis_delayed_traj;
                num = sum((xdisp - xpred).^2,'all');
                den = sum((xdisp - mean(xdisp,2)).^2,'all') + 1e-12;
                pop_predR2_vis(c,subj,tr) = 1 - num/den;
            end

            % Drive decomposition diagnostics
            if isfield(cache_full,'drive_rec_norm') && isfield(cache_full,'drive_in_norm') && isfield(cache_full,'drive_fb_norm')
                pop_drive_rec_rms(c,subj,tr) = sqrt(mean(cache_full.drive_rec_norm(:).^2));
                pop_drive_inp_rms(c,subj,tr) = sqrt(mean(cache_full.drive_in_norm(:).^2));
                pop_drive_fb_rms(c,subj,tr)  = sqrt(mean(cache_full.drive_fb_norm(:).^2));
            end

            % Learning update from FULL only
            grads = backward_trial(target_traj, context, params, cache_full, A_d, B_d, C_hand, M_vis, cfg_mr, cfg.betaF);
            grads = clip_gradients(grads, cfg_mr.clip_grad);

            Wrec_before = params.W_rec_ff;
            [params, dW_used] = apply_updates(params, grads, cache_full, cfg_mr, cond);
            pop_dWrec_rel(c,subj,tr) = norm(params.W_rec_ff - Wrec_before,'fro') / (norm(Wrec_before,'fro') + 1e-12);


            % Cumulative drift of Wrec from MR start
            pop_Wrec_drift(c,subj,tr) = norm(params.W_rec_ff - Wrec_ref_mr,'fro') / (norm(Wrec_ref_mr,'fro') + 1e-12);

            % Alignment between actual Delta Wrec and the local update direction
            if ~isempty(dW_used)
                dW_act = params.W_rec_ff - Wrec_before;
                num = sum(dW_act(:) .* dW_used(:));
                den = norm(dW_act(:)) * norm(dW_used(:)) + 1e-12;
                pop_Wrec_align(c,subj,tr) = num / den;
            end

            % Snapshot Wrec for low-rank analysis
            if mod(tr, snap_stride) == 0
                si = tr / snap_stride;
                if si >= 1 && si <= nSnaps
                    Wrec_snaps{c,subj}(:,:,si) = params.W_rec_ff;

                    % Snapshot teaching directions for mode projections:
                    %   local update is approximately dW ~ post * eligibility'
                    if isempty(post_snaps{c,subj})
                        post_snaps{c,subj} = zeros(Nff, nSnaps);
                        elig_snaps{c,subj} = zeros(Nff, nSnaps);
                    end

                    % post(t) = W_fb_ff * uF_raw(t), with the same delay as the local rule
                    post = params.W_fb_ff * cache_full.u_F_raw_hist;  % Nff x T
                    if cfg_mr.local_use_delay && cfg_mr.tau_steps_vis > 0
                        d = cfg_mr.tau_steps_vis;
                        post_d = zeros(size(post));
                        post_d(:, (d+1):end) = post(:, 1:(end-d));
                        post = post_d;
                    end
                    if isfield(cache_full,'valid_sample_mask')
                        post = post .* cache_full.valid_sample_mask;
                    end
                    post_dir = mean(post, 2);
                    if norm(post_dir) < 1e-12
                        post_dir = mean(abs(post), 2);
                    end
                    post_dir = post_dir / (norm(post_dir) + 1e-12);
                    post_snaps{c,subj}(:,si) = post_dir;

                    % eligibility(t): leaky trace of the previous h_ff state
                    h_ff_loc = cache_full.h_ff_hist;
                    Tloc = size(h_ff_loc,2);
                    r = zeros(Nff,1);
                    elig_alpha = cfg_mr.dt / cfg_mr.elig_tau_sec;
                    r_trace = zeros(Nff, Tloc);
                    for tt = 2:Tloc
                        r = (1 - elig_alpha) * r + elig_alpha * h_ff_loc(:, tt-1);
                        r_trace(:, tt) = r;
                    end
                    elig_dir = mean(r_trace, 2);
                    elig_dir = elig_dir / (norm(elig_dir) + 1e-12);
                    elig_snaps{c,subj}(:,si) = elig_dir;

                end
            end
            if mod(tr, 100) == 0
                fprintf('    MR: %d / %d\n', tr, nTrials_MR);
            end
        end

        %% 4) WASHOUT (probe, no learning)
        og_total_acc = zeros(1, nFreq);  % Accumulators per freq
        og_ff_acc = zeros(1, nFreq);
        og_fb_acc = zeros(1, nFreq);

        for tr = 1:nTrials_Wash
            rng(trialSeeds_wash_all(tr));
            target_traj = make_sum_of_sines_2D_xy(t_vec, freqs_x, freqs_y, amp_x, amp_y);

            noiseSeed = make_subj_noise_seed(trialSeeds_wash_all(tr), subj, 4);
            rng(noiseSeed);
            rng_state_before_forward = rng;

            context = 0;
            M_vis   = eye(2);

            cfg_w = cfg;
            % Allow learning during washout: update the feedback controller and recurrent FF weights
            cfg_w.ff_track_to_ff = 1.0;
            cfg_w.lambda_tutor   = 0.0;
            cfg_w.local_only_MR  = false;

            rng(rng_state_before_forward);
            [~, cache_full] = forward_trial(target_traj, context, params, A_d, B_d, C_hand, M_vis, ...
                true, cfg.betaF, use_ff, cfg_w);

            og_total = compute_orthogonal_gain_multifit(cache_full.hand_traj, target_traj, t_vec, freqs_x, freqs_y, 2, 'real');

            % Store washout OG per trial for decay plots
            pop_OG_total_wash_trials(c, subj, tr, :) = og_total;

            % FB-only closed-loop: feedback ON, FF OFF
            rng(rng_state_before_forward);
            [~, cache_FB_closed] = forward_trial(target_traj, context, params, A_d, B_d, C_hand, M_vis, ...
                true, cfg.betaF, false, cfg_w);

            og_fb_closed = compute_orthogonal_gain_multifit(cache_FB_closed.hand_traj, target_traj, t_vec, freqs_x, freqs_y, 2, 'real');

            % FF-only closed-loop: feedback OFF, FF ON
            rng(rng_state_before_forward);
            [~, cache_FF_closed] = forward_trial(target_traj, context, params, A_d, B_d, C_hand, M_vis, ...
                false, cfg.betaF, true, cfg_w);

            og_ff_closed = compute_orthogonal_gain_multifit(cache_FF_closed.hand_traj, target_traj, t_vec, freqs_x, freqs_y, 2, 'real');

            % Accumulate per freq
            og_total_acc = og_total_acc + og_total;
            og_ff_acc = og_ff_acc + og_ff_closed;
            og_fb_acc = og_fb_acc + og_fb_closed;

            
            % Learning update during washout (from FULL only)
            grads = backward_trial(target_traj, context, params, cache_full, A_d, B_d, C_hand, M_vis, cfg_w, cfg.betaF);
            grads = clip_gradients(grads, cfg_w.clip_grad);
            params = apply_updates(params, grads, cache_full, cfg_w, cond);

            fprintf('    Washout trial %d / %d done.\n', tr, nTrials_Wash);
        end  % This 'end' closes the 'for tr' loop

        % Average and store (outside tr loop, still inside cond/subj)
        pop_OG_total_wash(c, subj, :) = og_total_acc / nTrials_Wash;
        pop_OG_FF_closed_wash(c, subj, :) = og_ff_acc / nTrials_Wash;
        pop_OG_FB_closed_wash(c, subj, :) = og_fb_acc / nTrials_Wash;


        % After-effect = average washout minus baseline mean (per frequency)
        % Make everything 7x1 explicitly (prevents 7x7 implicit expansion)
        base_total = squeeze(pop_OG_total_base_mean(c,subj,:));  base_total = base_total(:);
        base_ffC   = squeeze(pop_OG_FF_closed_base_mean(c,subj,:)); base_ffC = base_ffC(:);
        base_fbC   = squeeze(pop_OG_FB_closed_base_mean(c,subj,:)); base_fbC = base_fbC(:);

        wash_total = squeeze(pop_OG_total_wash(c,subj,:));        wash_total = wash_total(:);
        wash_ffC   = squeeze(pop_OG_FF_closed_wash(c,subj,:));    wash_ffC   = wash_ffC(:);
        wash_fbC   = squeeze(pop_OG_FB_closed_wash(c,subj,:));    wash_fbC   = wash_fbC(:);

        base_total = max(min(base_total,1),-1);
        base_ffC   = max(min(base_ffC,  1),-1);
        base_fbC   = max(min(base_fbC,  1),-1);

        pop_AE_total(c,subj,:)     = reshape(wash_total - base_total, 1,1,[]);
        pop_AE_FF_closed(c,subj,:) = reshape(wash_ffC   - base_ffC,   1,1,[]);
        pop_AE_FB_closed(c,subj,:) = reshape(wash_fbC   - base_fbC,   1,1,[]);

        fprintf('    Washout done.\n');
    end

    fprintf('\nSubject %d done.\n', subj);
end

fprintf('\nAll subjects done.\n');

%% ------------------------------------------------------------------------
% PLOTTING
% ------------------------------------------------------------------------
for c = 1:nConds
    cond = cond_names{c};

    OGtot = squeeze(pop_OG_total(c,:,:,:));      % subj x trial x freq
    OGffC = squeeze(pop_OG_FF_closed(c,:,:,:));
    OGfbC = squeeze(pop_OG_FB_closed(c,:,:,:));

    title_str = sprintf('%s | vis=%.0fms prop=%.0fms |', ...
        cond, 1000*cfg.vis_delay_sec, 1000*cfg.proprio_delay_sec);


    baseTailTot = squeeze(pop_OG_total_base_buf(c,:,:,:));
    if ismatrix(baseTailTot)
        baseTailTot = reshape(baseTailTot, 1, size(baseTailTot,1), size(baseTailTot,2));
    end

    mrTot = squeeze(pop_OG_total(c,:,:,:));
    if ismatrix(mrTot)
        mrTot = reshape(mrTot, 1, size(mrTot,1), size(mrTot,2));
    end
    totWithBase = cat(2, baseTailTot, mrTot);
    
    % Append washout trials (context=0) to show decay back toward baseline
    washTot = squeeze(pop_OG_total_wash_trials(c,:,:,:));
    if ismatrix(washTot)
        washTot = reshape(washTot, 1, size(washTot,1), size(washTot,2));
    end
    totWithBase = cat(2, totWithBase, washTot);

    plot_total_with_baseline_tail(totWithBase, freqs_x, Wb_plot, nTrials_MR, ['Total OG (Baseline tail + MR + Washout): ' title_str]);

    generate_decomposed_plots_blue(OGtot, OGffC, OGfbC, freqs_x, ['CLOSED-LOOP: ' title_str]);

    plot_wrec_update(squeeze(pop_dWrec_rel(c,:,:)), sprintf('%s: Wrec update magnitude', title_str));

    plot_wrec_alignment(squeeze(pop_Wrec_align(c,:,:)), sprintf('%s: Wrec alignment with teaching signal', title_str));
    plot_wrec_lowrank_across_subjects(Wrec_snaps(c,:), Wrec_ref_store(c,:), snap_stride, sprintf('%s: Low-rank structure of Delta Wrec', title_str), post_snaps(c,:), elig_snaps(c,:));

    % Tracking error (baseline tail + MR)
    plot_error_baseline_then_mr(squeeze(pop_MSE_base_buf(c,:,:)), squeeze(pop_MSE_full(c,:,:)), Wb_plot, ...
        sprintf('%s: Tracking MSE (Baseline tail + MR)', title_str), nTrials_MR);
    %plot_drive_decomposition(squeeze(pop_drive_rec_rms(c,:,:)), squeeze(pop_drive_inp_rms(c,:,:)), ...
    %    sprintf('%s: FF drive decomposition', title_str));


    % After-effect (Washout minus BaselineMean) by frequency
    plot_aftereffect_by_freq( ...
        squeeze(pop_AE_total(c,:,:)), ...
        squeeze(pop_AE_FF_closed(c,:,:)), ...
        squeeze(pop_AE_FB_closed(c,:,:)), ...
        freqs_x, ...
        sprintf('%s: After-effect (Washout - BaselineMean)', title_str));

    % Early vs late MR trajectories (first 2.0 s)
    if ~isempty(traj_early{c,1}) && ~isempty(traj_late{c,1})
        plot_early_late_trajectories(traj_early{c,1}, traj_late{c,1}, cfg.dt, 2, ...
            sprintf('%s: Early vs Late MR trajectories', title_str));
    end

end



%% ========================================================================
% FORWARD PASS
% ========================================================================
function [loss, cache] = forward_trial(target_traj, context, params, A_d, B_d, C_hand, M_vis, use_feedback, betaF, use_ff, cfg)
[~, T] = size(target_traj);
N_ff = params.N_ff;

cache.target_traj = target_traj;
cache.context = context;

s_t = zeros(4,1);

cache.s_hist            = zeros(4, T+1);
cache.h_ff_hist         = zeros(N_ff, T);
cache.u_ff_hist         = zeros(2, T);
cache.u_fb_hist         = zeros(2, T);
cache.hand_traj         = zeros(2, T);
cache.cursor_true_traj  = zeros(2, T);
cache.cursor_disp_traj  = zeros(2, T);
cache.u_fb_contrib_hist = zeros(2, T);

cache.u_F_raw_hist      = zeros(params.k_F, T);
cache.h_F_hist          = zeros(params.N_F_hid, T);
cache.a_F_hist          = zeros(params.N_F_hid, T);
cache.y_F_hist          = zeros(params.D_in_F, T);

cache.z_ff_hist         = zeros(params.in_dim_ff, T);
cache.a_ff_hist         = zeros(N_ff, T);

cache.betaF = betaF;

cache.drive_rec_norm = zeros(1,T);
cache.drive_in_norm  = zeros(1,T);
cache.drive_fb_norm  = zeros(1,T);

cache.valid_sample_mask = ones(1,T);

cache.spe_vis_traj           = zeros(2,T);
cache.x_hat_vis_delayed_traj = zeros(2,T);

% Store GC activations 
cache.gc_hist = zeros(cfg.cb_Ngc, T);

% Tutor signal for FF 
cache.u_tutor_hist = zeros(2, T);

C_vis = M_vis * C_hand;
cache.C_vis = C_vis;

x_cursor_init = C_vis * s_t;

h_prev    = zeros(N_ff, 1);
a_ff_prev = zeros(N_ff, 1);

e_vis_lp    = zeros(2,1);
e_prop_lp   = zeros(2,1);
e_vis_tau_prev = zeros(2,1);
e_prop_tau_prev = zeros(2,1);

prev_x_star = zeros(2,1);

cache.s_hist(:,1) = s_t;

d_vis  = cfg.vis_delay_steps;
d_prop = cfg.proprio_delay_steps;
d_max  = max(d_vis, d_prop);

s_hat      = zeros(4,1);
s_hat_hist = repmat(s_hat, 1, d_max+1);

u_cmd_hist = zeros(2, max(d_max,1));

Kh = cfg.cb_Khist;
u_cb_hist = zeros(2, max(Kh,1));

x_disp_prev = x_cursor_init;

for t = 1:T
    x_star = target_traj(:,t);
    if t == 1
        v_star = [0; 0];
    else
        v_star = (x_star - prev_x_star) / cfg.dt;
    end

    x_hand        = C_hand * s_t;
    x_cursor_true = C_vis  * s_t;

    cache.hand_traj(:,t)        = x_hand;
    cache.cursor_true_traj(:,t) = x_cursor_true;

    idx_disp = t - d_vis;
    if idx_disp < 1
        x_disp = x_cursor_init;
    else
        x_disp = cache.cursor_true_traj(:, idx_disp);
    end

    if cfg.vis_meas_noise_std > 0
        x_disp = x_disp + cfg.vis_meas_noise_std * randn(2,1);
    end
    cache.cursor_disp_traj(:,t) = x_disp;

    if d_max >= 1
        s_hat_hist(:,2:end) = s_hat_hist(:,1:end-1);
        s_hat_hist(:,1)     = s_hat;
    else
        s_hat_hist(:,1)     = s_hat;
    end

    x_hat_vis_delayed = C_vis * s_hat_hist(:, d_vis+1);
    cache.x_hat_vis_delayed_traj(:,t) = x_hat_vis_delayed;

    idx_prop = t - d_prop;
    if idx_prop < 1
        x_prop = C_hand * zeros(4,1);
    else
        x_prop = cache.hand_traj(:, idx_prop);
    end
    if cfg.proprio_noise_std > 0
        x_prop = x_prop + cfg.proprio_noise_std * randn(2,1);
    end

    x_hat_prop_delayed = C_hand * s_hat_hist(:, d_prop+1);

    spe_vis  = x_disp - x_hat_vis_delayed;
    spe_prop = x_prop - x_hat_prop_delayed;

    spe_vis  = max(min(spe_vis,  cfg.spe_clip), -cfg.spe_clip);
    spe_prop = max(min(spe_prop, cfg.spe_clip), -cfg.spe_clip);

    cache.spe_vis_traj(:,t) = spe_vis;

    s_hat_hist(:, d_vis+1)  = s_hat_hist(:, d_vis+1)  + cfg.K_vis  * (C_vis'  * spe_vis);
    s_hat_hist(:, d_prop+1) = s_hat_hist(:, d_prop+1) + cfg.K_prop * (C_hand' * spe_prop);

    if cfg.use_cb_predictor
        uvec = reshape(u_cb_hist(:,1:Kh), [], 1);
        p_t  = [x_disp_prev; uvec; context];
        g_t  = max(0, params.cb.W_gc * p_t + params.cb.b_gc);

        cache.gc_hist(:,t) = g_t;  % Store GC features

        ds = params.cb.W_state * g_t;
        ds = max(min(ds, cfg.cb_ds_clip), -cfg.cb_ds_clip);

        s_hat_hist(:, d_vis+1) = s_hat_hist(:, d_vis+1) + ds;
        x_disp_prev = x_disp;
    else
        cache.gc_hist(:,t) = 0;
    end

    if d_max >= 1
        s_tmp = s_hat_hist(:, d_max+1);
        for k = d_max:-1:1
            uk    = u_cmd_hist(:, k);
            s_tmp = A_d * s_tmp + B_d * uk;
            s_hat_hist(:, k) = s_tmp;
        end
        s_hat = s_hat_hist(:,1);
    else
        s_hat = s_hat_hist(:,1);
    end

    x_hat_cursor_current = C_vis  * s_hat;
    x_hat_hand_current   = C_hand * s_hat;

    % A) Display-aligned target error: compare delayed target to delayed sensory
    idx_v = t - d_vis;   if idx_v < 1, idx_v = 1; end
    idx_p = t - d_prop;  if idx_p < 1, idx_p = 1; end

    x_star_v = target_traj(:, idx_v);   % target at visual delay time
    x_star_p = target_traj(:, idx_p);   % target at proprio delay time

    % Use the same sensory variables driving the controller:
    % x_disp is delayed visual cursor, x_prop is delayed proprioceptive hand position
    e_vis  = x_star_v - x_disp;
    e_prop = (M_vis') * x_star_p - x_prop;


    e_vis_lp  = (1 - cfg.alpha_lp) * e_vis_lp  + cfg.alpha_lp * e_vis;
    e_prop_lp = (1 - cfg.alpha_lp) * e_prop_lp + cfg.alpha_lp * e_prop;

    % predictor-based fast tracking errors (HF-capable)
    x_star_now = x_star;  % target at current time t
    e_vis_tau  = x_star_now - x_hat_cursor_current;                 % cursor-space
    e_prop_tau = (M_vis') * x_star_now - x_hat_hand_current;        % hand-space


    % derivative of innovation (HF-capable)
    de_vis  = (e_vis_tau  - e_vis_tau_prev)  / cfg.dt;
    de_prop = (e_prop_tau - e_prop_tau_prev) / cfg.dt;

    e_vis_tau_prev  = e_vis_tau;
    e_prop_tau_prev = e_prop_tau;

    y_F_t = [cfg.gain_vis_tau  * e_vis_tau;
        cfg.gain_vis_lp   * e_vis_lp;
        cfg.gain_vis_de   * de_vis;
        cfg.gain_prop_tau * e_prop_tau;
        cfg.gain_prop_lp  * e_prop_lp;
        cfg.gain_prop_de  * de_prop;
        context];

    cache.y_F_hist(:,t) = y_F_t;

    if use_feedback
        a_F_t = params.W_in_F * y_F_t + params.b_in_F;
        h_F_t = min(max(0, a_F_t), cfg.hF_clip);
        u_F_raw = params.W_out_F * h_F_t + params.b_out_F;
        u_fb    = params.W_fb_readout * u_F_raw + params.b_fb_readout;
    else
        a_F_t    = zeros(params.N_F_hid,1);
        h_F_t    = zeros(params.N_F_hid,1);
        u_F_raw  = zeros(params.k_F,1);
        u_fb     = zeros(2,1);
    end

    cache.a_F_hist(:,t)     = a_F_t;
    cache.h_F_hist(:,t)     = h_F_t;
    cache.u_F_raw_hist(:,t) = u_F_raw;

    % Tutor  (in the same units as u_ff)
    % Store feedback contribution (same units as u_ff). Tutor is built after the loop by delay-lead shifting.
    u_fb_contrib = betaF * u_fb;
    cache.u_fb_contrib_hist(:,t) = u_fb_contrib;


    z_ff_t = [x_star; v_star; context];
    cache.z_ff_hist(:,t) = z_ff_t;

    drive_rec = params.W_rec_ff * h_prev;
    drive_in  = params.W_in_ff  * z_ff_t;
    
    cache.drive_rec_norm(t) = norm(drive_rec);
    cache.drive_in_norm(t)  = norm(drive_in);
    cache.drive_fb_norm(t)  = 0;

    drive = drive_rec + drive_in + params.b_ff;
    a_ff_t = (1 - cfg.alpha_rnn) * a_ff_prev + cfg.alpha_rnn * drive;
    a_ff_prev = a_ff_t;

    h_t = min(max(0, a_ff_t), cfg.h_clip);

    u_ff = params.W_out_ff * h_t + params.b_out_ff;
    if ~use_ff
        u_ff = zeros(2,1);
    end

    u_ff = max(min(u_ff, cfg.u_clip), -cfg.u_clip);
    u_fb = max(min(u_fb, cfg.u_clip), -cfg.u_clip);

    if cfg.motor_noise_std > 0
        noise = cfg.motor_noise_std * randn(2,1);
    else
        noise = zeros(2,1);
    end

    u_cmd = u_ff + betaF * u_fb;
    u_cmd = max(min(u_cmd, cfg.u_clip), -cfg.u_clip);

    u_t = u_cmd + noise;
    u_t = max(min(u_t, cfg.u_clip), -cfg.u_clip);

    s_next = A_d * s_t + B_d * u_t;
    if cfg.use_state_clip
        s_next = clamp_state(s_next, cfg.x_clip, cfg.v_clip);
    end

    s_hat = A_d * s_hat + B_d * u_cmd;
    if cfg.use_state_clip
        s_hat = clamp_state(s_hat, cfg.x_clip, cfg.v_clip);
    end

    if d_max >= 1
        u_cmd_hist(:,2:end) = u_cmd_hist(:,1:end-1);
        u_cmd_hist(:,1)     = u_cmd;
    else
        u_cmd_hist(:,1)     = u_cmd;
    end
    if Kh >= 1
        u_cb_hist(:,2:end) = u_cb_hist(:,1:end-1);
        u_cb_hist(:,1)     = u_cmd;
    else
        u_cb_hist(:,1)     = u_cmd;
    end

    cache.s_hist(:,t+1)   = s_next;
    cache.h_ff_hist(:,t)  = h_t;
    cache.a_ff_hist(:,t)  = a_ff_t;
    cache.u_ff_hist(:,t)  = u_ff;
    cache.u_fb_hist(:,t)  = u_fb;

    s_t = s_next;
    h_prev = h_t;
    prev_x_star = x_star;
end

cache.total_disp_delay_steps = d_vis;
cache.T_eff = max(1, T - d_vis);
cache.track_grad_scale = 1.0;

loss = 0.0;

% Build delay-lead tutor
dlead = cfg.vis_delay_steps;
if isfield(cfg,'tutor_lead_steps')
    dlead = cfg.tutor_lead_steps;
end
dlead = max(0, round(dlead));

cache.u_tutor_hist = zeros(2, T);
if T > dlead
    cache.u_tutor_hist(:, 1:(T-dlead)) = cache.u_fb_contrib_hist(:, (1+dlead):T);
end

end


%% ========================================================================
% BACKWARD PASS
% ========================================================================
function grads = backward_trial(target_traj, context, params, cache, A_d, B_d, C_hand, M_vis, cfg, betaF) %#ok<INUSD>
[~, T] = size(target_traj);
N_ff = params.N_ff;

s_hist            = cache.s_hist;
h_ff_hist         = cache.h_ff_hist;
u_ff_hist         = cache.u_ff_hist;
u_fb_hist         = cache.u_fb_hist;
cursor_disp_traj  = cache.cursor_disp_traj;
z_ff_hist         = cache.z_ff_hist;
y_F_hist          = cache.y_F_hist;
h_F_hist          = cache.h_F_hist;
u_F_raw_hist      = cache.u_F_raw_hist;
a_ff_hist         = cache.a_ff_hist;
a_F_hist          = cache.a_F_hist;

C_vis = cache.C_vis;

total_delay_steps = cache.total_disp_delay_steps;
T_eff = cache.T_eff;

u_tutor_hist = cache.u_tutor_hist;
valid_sample_mask    = cache.valid_sample_mask;

dlead = cfg.vis_delay_steps;
if isfield(cfg,'tutor_lead_steps')
    dlead = cfg.tutor_lead_steps;
end
dlead = max(0, round(dlead));

grads.W_rec_ff     = zeros(size(params.W_rec_ff));
grads.W_in_ff      = zeros(size(params.W_in_ff));
grads.b_ff         = zeros(size(params.b_ff));
grads.W_out_ff     = zeros(size(params.W_out_ff));
grads.b_out_ff     = zeros(size(params.b_out_ff));
grads.W_fb_ff      = zeros(size(params.W_fb_ff));

grads.W_in_F       = zeros(size(params.W_in_F));
grads.b_in_F       = zeros(size(params.b_in_F));
grads.W_out_F      = zeros(size(params.W_out_F));
grads.b_out_F      = zeros(size(params.b_out_F));
grads.W_fb_readout = zeros(size(params.W_fb_readout));
grads.b_fb_readout = zeros(size(params.b_fb_readout));

alpha_track = (1 / T_eff) * cache.track_grad_scale;

gamma_ff    = 2 * cfg.lambda_u_ff / numel(u_ff_hist);
gamma_fb    = 2 * cfg.lambda_u_fb * (betaF^2) / numel(u_fb_hist);
gamma_du_ff = 2 * cfg.lambda_du_ff / numel(u_ff_hist);

% Tutor loss weight (squared error on u_ff - u_tutor)
gamma_tutor = 2 * cfg.lambda_tutor / numel(u_ff_hist);

g_h_next = zeros(N_ff,1);
g_s_next = zeros(4,1);
g_a_next = zeros(N_ff,1);

for t = T:-1:1
    s_t     = s_hist(:,t);
    u_ff_t  = u_ff_hist(:,t);
    u_fb_t  = u_fb_hist(:,t);
    h_t     = h_ff_hist(:,t);
    z_t     = z_ff_hist(:,t);

    h_F_t     = h_F_hist(:,t);
    u_F_raw_t = u_F_raw_hist(:,t);
    y_F_t     = y_F_hist(:,t);

    a_ff_t = a_ff_hist(:,t);
    a_F_t  = a_F_hist(:,t);

    if t > 1
        h_prev = h_ff_hist(:, t-1);
    else
        h_prev = zeros(N_ff, 1);
    end

    % Tracking loss uses DISPLAY cursor at display-time (t+delay)
    if t <= (T - total_delay_steps)
        e_g = target_traj(:, t + total_delay_steps) - cursor_disp_traj(:, t + total_delay_steps);
        dL_dx = -alpha_track * e_g;
    else
        dL_dx = [0; 0];
    end

    if t <= (T - total_delay_steps)
        tj = t + total_delay_steps;
        sample_weight = valid_sample_mask(tj);
        dL_dx = sample_weight * dL_dx;
    end


    x_hand = C_hand * s_t;
    dL_dx_hand = (2 * cfg.lambda_pos / T) * x_hand;

    g_s_track  = C_vis' * dL_dx + C_hand' * dL_dx_hand;
    g_s_future = A_d' * g_s_next;
    g_s_t = g_s_track + g_s_future;

    dL_du_dyn = B_d' * g_s_next;

    % -------------------------------
    % During MR, FF learns from the tutor signal instead.
    % -------------------------------
    dL_du_ff = cfg.ff_track_to_ff * dL_du_dyn + gamma_ff * u_ff_t;

    % Tutor imitation loss 
    tj = t + dlead;
    sample_weight = 1.0;
    if cfg.use_valid_sample_mask && tj <= numel(valid_sample_mask)
        sample_weight = valid_sample_mask(tj);
    end

    if cfg.lambda_tutor > 0 && t <= (T - dlead)
        dL_du_ff = dL_du_ff + sample_weight * gamma_tutor * (u_ff_t - u_tutor_hist(:,t));
    end

    if cfg.lambda_du_ff > 0
        if t == 1
            u_next = u_ff_hist(:, t+1);
            dL_du_ff = dL_du_ff + gamma_du_ff * (u_ff_t - u_next);
        elseif t == T
            u_prev = u_ff_hist(:, t-1);
            dL_du_ff = dL_du_ff + gamma_du_ff * (u_ff_t - u_prev);
        else
            u_prev = u_ff_hist(:, t-1);
            u_next = u_ff_hist(:, t+1);
            dL_du_ff = dL_du_ff + gamma_du_ff * (2*u_ff_t - u_prev - u_next);
        end
    end

    % Feedback still receives tracking gradients (keeps real-time control)
    dL_du_fb = betaF * dL_du_dyn + gamma_fb * u_fb_t;

    grads.W_out_ff = grads.W_out_ff + dL_du_ff * h_t';
    grads.b_out_ff = grads.b_out_ff + dL_du_ff;

    g_h_from_out = params.W_out_ff' * dL_du_ff;
    g_h_t = g_h_from_out + g_h_next;

    mask_ff = (a_ff_t > 0) & (a_ff_t < cfg.h_clip);
    g_a_immediate = g_h_t .* mask_ff;

    g_a_from_next = (1 - cfg.alpha_rnn) * g_a_next;
    g_a_t = g_a_immediate + g_a_from_next;

    g_drive = cfg.alpha_rnn * g_a_t;

    grads.W_rec_ff = grads.W_rec_ff + g_drive * h_prev';
    grads.W_in_ff  = grads.W_in_ff  + g_drive * z_t';
    grads.b_ff     = grads.b_ff     + g_drive;
    

    g_h_prev_from_rec = params.W_rec_ff' * g_drive;

    grads.W_fb_readout = grads.W_fb_readout + dL_du_fb * u_F_raw_t';
    grads.b_fb_readout = grads.b_fb_readout + dL_du_fb;

    
    g_uF_from_readout = params.W_fb_readout' * dL_du_fb;
    g_uF_total = g_uF_from_readout;

    grads.W_out_F = grads.W_out_F + g_uF_total * h_F_t';
    grads.b_out_F = grads.b_out_F + g_uF_total;

    g_hF_t = params.W_out_F' * g_uF_total;
    mask_F = (a_F_t > 0) & (a_F_t < cfg.hF_clip);
    g_a_F = g_hF_t .* mask_F;

    grads.W_in_F = grads.W_in_F + g_a_F * y_F_t';
    grads.b_in_F = grads.b_in_F + g_a_F;

    g_h_next = g_h_prev_from_rec;
    g_s_next = g_s_t;
    g_a_next = g_a_t;
end
end


%% ========================================================================
% PARAMETER UPDATES
% ========================================================================
function [params, dW_used] = apply_updates(params, grads, cache, cfg, cond)
dW_used = [];
% Feedback NN
params.W_in_F       = params.W_in_F       - cfg.lr_F_adapt * grads.W_in_F;
params.b_in_F       = params.b_in_F       - cfg.lr_F_adapt * grads.b_in_F;
params.W_out_F      = params.W_out_F      - cfg.lr_F_adapt * grads.W_out_F;
params.b_out_F      = params.b_out_F      - cfg.lr_F_adapt * grads.b_out_F;
params.W_fb_readout = params.W_fb_readout - cfg.lr_F_adapt * grads.W_fb_readout;
params.b_fb_readout = params.b_fb_readout - cfg.lr_F_adapt * grads.b_fb_readout;

% FF non-recurrent
params.W_out_ff = params.W_out_ff - cfg.lr_base    * grads.W_out_ff;
params.b_out_ff = params.b_out_ff - cfg.lr_base    * grads.b_out_ff;
params.W_in_ff  = params.W_in_ff  - cfg.lr_base    * grads.W_in_ff;
params.b_ff     = params.b_ff     - cfg.lr_base    * grads.b_ff;

% Recurrent weights by condition
if strcmp(cond, 'BPTT')
    params.W_rec_ff = params.W_rec_ff - cfg.lr_base * grads.W_rec_ff;
elseif strcmp(cond, 'LocalFeedback')
    [params.W_rec_ff, dW_used] = update_Wrec_LocalFeedbackRule(params.W_rec_ff, cache, params, cfg);
elseif strcmp(cond, 'ERRLOCAL')
    params.W_rec_ff = update_Wrec_ErrorDrivenLocal(params.W_rec_ff, cache, params, cfg);
else
    error('Unknown condition: %s', cond);
end

params = sanitize_params(params);

% CB predictor update
if cfg.use_cb_predictor && isfield(cache,'spe_vis_traj') && isfield(cache,'C_vis')
    spe = cache.spe_vis_traj;
    C_vis = cache.C_vis;

    if isfield(cache,'gc_hist')
        G = cache.gc_hist;
    else        
        params.W_rec_ff = enforce_spectral_radius_det(params.W_rec_ff, cfg.target_rho, cfg.rho_power_iters);
        return;
    end

    mask = ones(1, size(spe,2));
    if isfield(cache,'valid_sample_mask')
        mask = cache.valid_sample_mask;
    end
    idx = find(mask > 0.5);

    if ~isempty(idx)
        spe_m = spe(:, idx);
        Gm    = G(:, idx);

        e_state = C_vis' * spe_m;
        dW_state = (e_state * Gm') / max(1, numel(idx));
        params.cb.W_state = params.cb.W_state + cfg.lr_cb_pred * dW_state;
        params.cb.W_state(~isfinite(params.cb.W_state)) = 0;
    end
end

params.W_rec_ff = enforce_spectral_radius_det(params.W_rec_ff, cfg.target_rho, cfg.rho_power_iters);
end

function params = sanitize_params(params)
f = fieldnames(params);
for i = 1:numel(f)
    X = params.(f{i});
    if isnumeric(X)
        X(~isfinite(X)) = 0;
        params.(f{i}) = X;
    end
end
if isfield(params,'cb')
    g = fieldnames(params.cb);
    for i = 1:numel(g)
        X = params.cb.(g{i});
        if isnumeric(X)
            X(~isfinite(X)) = 0;
            params.cb.(g{i}) = X;
        end
    end
end
end

function grads = clip_gradients(grads, clip_val)
f = fieldnames(grads);
for i = 1:numel(f)
    g = grads.(f{i});
    g(~isfinite(g)) = 0;
    grads.(f{i}) = g;
end

g_norm2 = 0;
for i = 1:numel(f)
    g_norm2 = g_norm2 + sum(grads.(f{i})(:).^2);
end
g_norm = sqrt(g_norm2);

if ~isfinite(g_norm) || g_norm <= 0
    return;
end

if g_norm > clip_val
    scale = clip_val / g_norm;
    for i = 1:numel(f)
        grads.(f{i}) = grads.(f{i}) * scale;
    end
end
end

%% ========================================================================
% PARAMETER INITIALIZATION
% ========================================================================
function params = init_params_ff_F(N_ff, in_dim_ff, N_F_hid, k_F, D_in_F, cfg)
params.N_ff      = N_ff;
params.in_dim_ff = in_dim_ff;
params.N_F_hid   = N_F_hid;
params.k_F       = k_F;
params.D_in_F    = D_in_F;

params.W_rec_ff = 0.9/sqrt(N_ff) * randn(N_ff);
params.W_in_ff  = 1/sqrt(in_dim_ff) * randn(N_ff, in_dim_ff);
params.b_ff     = zeros(N_ff,1);

params.W_out_ff = 0.1/sqrt(N_ff) * randn(2, N_ff);
params.b_out_ff = zeros(2,1);

params.W_fb_ff  = 0.8/sqrt(k_F) * randn(N_ff, k_F);

params.W_in_F   = 0.6/sqrt(D_in_F) * randn(N_F_hid, D_in_F);
params.b_in_F   = zeros(N_F_hid,1);
params.W_out_F  = 0.15/sqrt(N_F_hid) * randn(k_F, N_F_hid);
params.b_out_F  = zeros(k_F,1);

params.W_fb_readout = 0.20 * randn(2, k_F);
params.b_fb_readout = zeros(2,1);

% CB predictor (GC features + learned W_state)
params.cb = struct();
D_pf = 2 + 2*cfg.cb_Khist + 1;
params.cb.W_gc   = (1/sqrt(D_pf)) * randn(cfg.cb_Ngc, D_pf);
params.cb.b_gc   = zeros(cfg.cb_Ngc, 1);
params.cb.W_state = 0.001 * randn(4, cfg.cb_Ngc);

% Local rule projector for ERRLOCAL
params.F_err = 0.1 / sqrt(2) * randn(params.N_ff, 2);
end

%% ========================================================================
% LOCAL RECURRENT-WEIGHT RULES
% ========================================================================
function [W_rec, dW_used] = update_Wrec_LocalFeedbackRule(W_rec, cache, params, cfg)
dW_used = [];
if cfg.local_only_MR
    if ~isfield(cache,'context') || cache.context ~= 1
        return;
    end
end

h_ff = cache.h_ff_hist;
uF   = cache.u_F_raw_hist;
[N_ff, T] = size(h_ff);

post = params.W_fb_ff * uF;

if cfg.local_use_delay && cfg.tau_steps_vis > 0
    d = cfg.tau_steps_vis;
    post_d = zeros(N_ff, T);
    post_d(:, (d+1):end) = post(:, 1:(end-d));
    post = post_d;
end

if isfield(cache,'valid_sample_mask')
    m = cache.valid_sample_mask;          % 1 x T
    post = post .* m;
end

r_trace = zeros(N_ff, T);
r = zeros(N_ff, 1);
elig_alpha = cfg.dt / cfg.elig_tau_sec;

for t = 2:T
    r = (1 - elig_alpha) * r + elig_alpha * h_ff(:, t-1);
    r_trace(:, t) = r;
end

stride = cfg.local_stride;
dW = zeros(N_ff, N_ff);
nSteps = 0;
for t = stride:stride:T
    dW = dW + post(:,t) * r_trace(:,t)';
    nSteps = nSteps + 1;
end
if nSteps < 1, return; end

dW = dW / (2 * nSteps);
dW_used = cfg.lr_RNN_local * dW;
W_rec = W_rec + dW_used;
end

function W_rec = update_Wrec_ErrorDrivenLocal(W_rec, cache, params, cfg)
if cfg.local_only_MR
    if ~isfield(cache,'context') || cache.context ~= 1
        return;
    end
end

eps = cache.spe_vis_traj;  % 2 x T
T = size(eps,2);
N = size(W_rec,1);

eps_d = eps;
if cfg.local_use_delay && cfg.tau_steps_vis > 0
    d = cfg.tau_steps_vis;
    eps_d = zeros(2, T);
    eps_d(:, (d+1):end) = eps(:, 1:(end-d));
end

if isfield(cache,'valid_sample_mask')
    m = cache.valid_sample_mask;
    eps_d = eps_d .* m;
end
post = params.F_err * eps_d;

h = cache.h_ff_hist;
r_trace = zeros(N, T);
r = zeros(N,1);
elig_alpha = cfg.dt / cfg.elig_tau_sec;

for t = 2:T
    r = (1 - elig_alpha) * r + elig_alpha * h(:,t-1);
    r_trace(:,t) = r;
end

stride = cfg.local_stride;
dW = zeros(N, N);
nSteps = 0;
for t = stride:stride:T
    dW = dW + post(:,t) * r_trace(:,t)';
    nSteps = nSteps + 1;
end
if nSteps < 1, return; end

dW = dW / (2 * nSteps);
W_rec = W_rec + cfg.lr_RNN_local * dW;
end

%% ========================================================================
% HELPERS
% ========================================================================
function seed = make_subj_noise_seed(baseSeed, subj, blockId)
M = uint64(2^32 - 1);
seed64 = mod(uint64(baseSeed) + uint64(100000)*uint64(subj) + uint64(10000000)*uint64(blockId), M);
seed = double(seed64);
if seed <= 0, seed = 1; end
end

function s = clamp_state(s, x_clip, v_clip)
s(1) = max(min(s(1), x_clip), -x_clip);
s(3) = max(min(s(3), x_clip), -x_clip);
s(2) = max(min(s(2), v_clip), -v_clip);
s(4) = max(min(s(4), v_clip), -v_clip);
end

function W = enforce_spectral_radius_det(W, target_rho, iters)
n = size(W,1);
v = ones(n,1);
nv = norm(v);
if nv < 1e-12, return; end
v = v / nv;

for k = 1:iters
    v = W * v;
    nv = norm(v);
    if nv < 1e-12, break; end
    v = v / nv;
end

rho = norm(W * v);
if ~isfinite(rho) || rho <= 0, return; end
if rho > target_rho
    W = (target_rho / rho) * W;
end
end

function traj = simulate_trajectory(u_seq, A_d, B_d, C_hand, cfg)
[~, T] = size(u_seq);
s_t = zeros(4,1);
traj = zeros(2, T);
for t = 1:T
    u_t = max(min(u_seq(:, t), cfg.u_clip), -cfg.u_clip);
    traj(:, t) = C_hand * s_t;
    s_next = A_d * s_t + B_d * u_t;
    if cfg.use_state_clip
        s_next = clamp_state(s_next, cfg.x_clip, cfg.v_clip);
    end
    s_t = s_next;
end
end

function [A_d, B_d, C] = build_arm_dynamics(dt, omega_n, zeta, k_gain)
omega2 = omega_n^2;
A_c = [0 1 0 0;
    -omega2 -2*zeta*omega_n 0 0;
    0 0 0 1;
    0 0 -omega2 -2*zeta*omega_n];
B_c = [0 0;
    k_gain 0;
    0 0;
    0 k_gain];
A_d = eye(4) + dt * A_c;
B_d = dt * B_c;
C   = [1 0 0 0;
    0 0 1 0];
end

function target_traj = make_sum_of_sines_2D_xy(t_vec, freqs_x, freqs_y, amp_x, amp_y)
nFreq = numel(freqs_x);
phi_x = 2*pi*rand(1, nFreq);
phi_y = 2*pi*rand(1, nFreq);

x_traj = zeros(1, numel(t_vec));
y_traj = zeros(1, numel(t_vec));
for k = 1:nFreq
    x_traj = x_traj + amp_x(k) * sin(2*pi*freqs_x(k) * t_vec + phi_x(k));
    y_traj = y_traj + amp_y(k) * sin(2*pi*freqs_y(k) * t_vec + phi_y(k));
end
target_traj = [x_traj; y_traj];
end

function og_vec = compute_orthogonal_gain_multifit(hand_traj, target_traj, t_vec, freqs_x, freqs_y, detrend_order, mode)
t = t_vec(:);
T = numel(t);

Hx = detrend_poly(hand_traj(1,:).', t, detrend_order);
Hy = detrend_poly(hand_traj(2,:).', t, detrend_order);
Tx = detrend_poly(target_traj(1,:).', t, detrend_order);
Ty = detrend_poly(target_traj(2,:).', t, detrend_order);

freqs_all = unique([freqs_x(:); freqs_y(:)]).';
nF = numel(freqs_all);

Phi = zeros(T, 2*nF);
for i = 1:nF
    w = 2*pi*freqs_all(i);
    Phi(:,2*i-1) = sin(w*t);
    Phi(:,2*i)   = cos(w*t);
end

bHx = Phi \ Hx;
bHy = Phi \ Hy;
bTx = Phi \ Tx;
bTy = Phi \ Ty;

toC = @(a,b) (b - 1i*a);

CxH = zeros(1,nF);
CyH = zeros(1,nF);
CxT = zeros(1,nF);
CyT = zeros(1,nF);
for i = 1:nF
    CxH(i) = toC(bHx(2*i-1), bHx(2*i));
    CyH(i) = toC(bHy(2*i-1), bHy(2*i));
    CxT(i) = toC(bTx(2*i-1), bTx(2*i));
    CyT(i) = toC(bTy(2*i-1), bTy(2*i));
end

og_vec = zeros(1, numel(freqs_x));
for k = 1:numel(freqs_x)
    ix = find(abs(freqs_all - freqs_x(k)) < 1e-12, 1);
    iy = find(abs(freqs_all - freqs_y(k)) < 1e-12, 1);

    Tx_fx = CxT(ix);
    Ty_fy = CyT(iy);

    if abs(Tx_fx) < 1e-9, Tx_fx = 1e-9; end
    if abs(Ty_fy) < 1e-9, Ty_fy = 1e-9; end

    G_xx = CxH(ix) / Tx_fx;
    G_yx = CyH(ix) / Tx_fx;
    G_xy = CxH(iy) / Ty_fy;
    G_yy = CyH(iy) / Ty_fy;

    if strcmp(mode,'real')
        og_vec(k) = -0.5*(real(G_xx) - real(G_xy) - real(G_yx) + real(G_yy));        
    else
        og_vec(k) = -0.5*(abs(G_xx) - abs(G_xy) - abs(G_yx) + abs(G_yy));
    end
end
end

function y_dt = detrend_poly(y, t, order)
if order <= 0
    y_dt = y - mean(y);
    return;
end
T = numel(t);
X = zeros(T, order+1);
for p = 0:order
    X(:, p+1) = t.^p;
end
b = X \ y;
y_dt = y - X*b;
end

%% ----------------- Plotting helpers ------------------------------------
function generate_decomposed_plots_blue(pop_OG_total, pop_OG_FF, pop_OG_FB, freqs, title_str)
if ismatrix(pop_OG_total)
    mu_Total = pop_OG_total;
    mu_FF    = pop_OG_FF;
    mu_FB    = pop_OG_FB;
    sem_Total = zeros(size(mu_Total));
    sem_FF    = zeros(size(mu_FF));
    sem_FB    = zeros(size(mu_FB));
else
    N = size(pop_OG_total, 1);
    mu_Total = squeeze(mean(pop_OG_total, 1));
    mu_FF    = squeeze(mean(pop_OG_FF, 1));
    mu_FB    = squeeze(mean(pop_OG_FB, 1));
    sem_Total = squeeze(std(pop_OG_total, 0, 1)) ./ sqrt(N);
    sem_FF    = squeeze(std(pop_OG_FF,    0, 1)) ./ sqrt(N);
    sem_FB    = squeeze(std(pop_OG_FB,    0, 1)) ./ sqrt(N);
end

nFreq = numel(freqs);
colors = gradual_blue(nFreq);

t_vec = 1:size(mu_Total, 1);
leg_strs = arrayfun(@(f) sprintf('%.2f Hz', f), freqs, 'UniformOutput', false);

figure('Color','w', 'Position', [100 100 500 380]); hold on;
for k = 1:nFreq
    plot_shaded(t_vec, mu_Total(:,k), sem_Total(:,k), colors(k,:));
end
yline(0, 'k--');
ylim([-1 1]); xlim([1 max(t_vec)]);
xlabel('Trials'); ylabel('Total Orthogonal Gain');
title(['Total OG: ' title_str], 'Interpreter','none');
grid on;
legend(leg_strs, 'Location', 'southeast', 'FontSize', 8);

figure('Color','w', 'Position', [650 100 500 380]);
colors_FF = gradual_purple(nFreq);
title(['Feedforward Component: ' title_str], 'Interpreter','none');
for k = 1:nFreq
    plot_shaded(t_vec, mu_FF(:,k), sem_FF(:,k), colors_FF(k,:)); hold on
end
yline(0, 'k--'); ylim([-1 1]); xlim([1 max(t_vec)]); grid on;
xlabel('Trials'); ylabel('OG');
legend(leg_strs, 'Location', 'southeast', 'FontSize', 8);

figure('Color','w', 'Position', [650 100 500 380]);
colors_FB = gradual_green(nFreq);
title(['Feedback Component: ' title_str], 'Interpreter','none');
for k = 1:nFreq
    plot_shaded(t_vec, mu_FB(:,k), sem_FB(:,k), colors_FB(k,:)); hold on
end
yline(0, 'k--'); ylim([-1 1]); xlim([1 max(t_vec)]); grid on;
xlabel('Trials'); ylabel('OG');
legend(leg_strs, 'Location', 'southeast', 'FontSize', 8);
end

function colors = gradual_blue(nFreq)
% Dark for LF (index 1), light for HF (index nFreq)
dark  = [0 0 1];
light = [0 1 1];

colors = zeros(nFreq,3);
for i = 1:nFreq
    a = (i-1) / max(1, nFreq-1);      % 0 (LF) -> 1 (HF)
    colors(i,:) = dark*(1-a) + light*a;
end
end

function colors = gradual_green(nFreq)
% Dark green for LF (index 1), bright green for HF (index nFreq)
dark  = [0.00 0.35 0.00];   % darker green
light = [0.00 1.00 0.00];   % bright green

colors = zeros(nFreq,3);
for i = 1:nFreq
    a = (i-1) / max(1, nFreq-1);      % 0 (LF) -> 1 (HF)
    colors(i,:) = dark*(1-a) + light*a;
end
end

function colors = gradual_purple(nFreq)
% Purple for LF (index 1), brighter purple for HF (index nFreq)
dark  = [0.45 0.00 0.55];   % darker purple
light = [0.85 0.35 1.00];   % brighter purple (lighter + more vivid)

colors = zeros(nFreq,3);
for i = 1:nFreq
    a = (i-1) / max(1, nFreq-1);      % 0 (LF) -> 1 (HF)
    colors(i,:) = dark*(1-a) + light*a;
end
end


function plot_shaded(x, mu, sem, c)
doFill = any(abs(sem(:)) > 0);
if doFill
    fill([x(:); flipud(x(:))], [mu(:)+sem(:); flipud(mu(:)-sem(:))], c, ...
        'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');
end
plot(x, mu, 'Color', c, 'LineWidth', 1.5);
end


function plot_wrec_update(dWrel, title_str)
dWrel = ensure_subj_by_trials(dWrel);
mu = mean(smoothdata(dWrel,"gaussian",10),1);
if size(dWrel,1) >= 2
    sem = std(dWrel,0,1)/sqrt(size(dWrel,1));
else
    sem = zeros(size(mu));
end
t = 1:numel(mu);

figure('Color','w','Position',[100 100 520 360]); hold on;
fill([t, fliplr(t)], [mu+sem, fliplr(mu-sem)], 'b', 'FaceAlpha',0.15,'EdgeColor','none');
plot(t, smoothdata(mu,"gaussian",10), 'b', 'LineWidth',2);
xlabel('MR trials'); ylabel('Relative ||dWrec|| (Fro)');
title(title_str, 'Interpreter','none'); grid on;
end



function plot_wrec_alignment(align, title_str)
align = ensure_subj_by_trials(align);
align = smoothdata(align, 'gaussian',20);

mu = mean(align,1,'omitnan');
if size(align,1) >= 2
    sem = std(align,0,1,'omitnan') / sqrt(size(align,1));
else
    sem = zeros(size(mu));
end
t = 1:numel(mu);

figure('Color','w','Position',[100 100 520 360]); hold on;
fill([t, fliplr(t)], [mu+sem, fliplr(mu-sem)], 'b', 'FaceAlpha',0.15,'EdgeColor','none');
plot(t, mu, 'b', 'LineWidth',2);
yline(0,'k--');
xlabel('MR trials'); ylabel('Cosine alignment');
title(title_str, 'Interpreter','none'); grid on;
ylim([-1 1]);
end

function plot_wrec_lowrank_across_subjects(Wrec_snaps_row, Wrec_ref_row, snap_stride, title_str, post_snaps_row, elig_snaps_row)
% Wrec_snaps_row : 1 x nSubjects cell, each Nff x Nff x nSnaps
% Wrec_ref_row   : 1 x nSubjects cell, each Nff x Nff
% post_snaps_row : 1 x nSubjects cell, each Nff x nSnaps (optional)
% elig_snaps_row : 1 x nSubjects cell, each Nff x nSnaps (optional)
%
% Local recurrent update is approximately: dW ~ post * eligibility'
% So we project:
%   Left singular vectors (U) onto post
%   Right singular vectors (V) onto eligibility

Klist = 1:5;

% Find number of snapshots from first non-empty
nSnaps = [];
for s = 1:numel(Wrec_snaps_row)
    if ~isempty(Wrec_snaps_row{s})
        nSnaps = size(Wrec_snaps_row{s}, 3);
        break;
    end
end
if isempty(nSnaps) || nSnaps < 2
    return;
end

nSubj = numel(Wrec_snaps_row);
nK = numel(Klist);

% cumulative (relative to Wref)
frac_cum = nan(nSubj, nSnaps, nK);

% incremental (between snapshots)
frac_inc = nan(nSubj, nSnaps, nK);
inc_norm = nan(nSubj, nSnaps);

% optional projections of top modes
doProj = (nargin >= 6) && ~isempty(post_snaps_row) && ~isempty(elig_snaps_row);
projU = nan(nSubj, nSnaps, nK);   % U (left) vs post
projV = nan(nSubj, nSnaps, nK);   % V (right) vs eligibility

for subj = 1:nSubj
    Wsn  = Wrec_snaps_row{subj};
    Wref = Wrec_ref_row{subj};

    if isempty(Wsn) || isempty(Wref)
        continue;
    end

    for si = 1:nSnaps
        dW = Wsn(:,:,si) - Wref;

        % Cumulative low-rank fraction
        svals = svd(dW);
        total = sum(svals.^2) + 1e-12;
        for kk = 1:nK
            k = Klist(kk);
            frac_cum(subj, si, kk) = sum(svals(1:min(k,end)).^2) / total;
        end

        % Optional projections of top modes onto teaching directions
        if doProj
            hasPost = ~isempty(post_snaps_row{subj}) && size(post_snaps_row{subj},2) >= si;
            hasElig = ~isempty(elig_snaps_row{subj}) && size(elig_snaps_row{subj},2) >= si;
            if hasPost && hasElig
                post = post_snaps_row{subj}(:,si);
                elig = elig_snaps_row{subj}(:,si);
                post = post / (norm(post) + 1e-12);
                elig = elig / (norm(elig) + 1e-12);

                [U,~,V] = svd(dW, 'econ');
                kMax = min(nK, size(U,2));
                projU(subj, si, 1:kMax) = abs(U(:,1:kMax)' * post);
                projV(subj, si, 1:kMax) = abs(V(:,1:kMax)' * elig);
            end
        end

        % Incremental low-rank fraction and norm
        if si >= 2
            dW_inc = Wsn(:,:,si) - Wsn(:,:,si-1);
            s2 = svd(dW_inc);
            total2 = sum(s2.^2) + 1e-12;
            for kk = 1:nK
                k = Klist(kk);
                frac_inc(subj, si, kk) = sum(s2(1:min(k,end)).^2) / total2;
            end
            inc_norm(subj, si) = norm(dW_inc,'fro') / (norm(Wsn(:,:,si-1),'fro') + 1e-12);
        end
    end
end

t = (1:nSnaps) * snap_stride;

% ------------------- Projections onto teaching directions -------------------
if doProj
    muU  = squeeze(mean(smoothdata(projU, "gaussian",10), 1, 'omitnan'));              % nSnaps x K
    muV  = squeeze(mean(smoothdata(projV, "gaussian",10), 1, 'omitnan'));              % nSnaps x K
    semU = squeeze(std(smoothdata(projU, "gaussian",10), 0, 1, 'omitnan')) / sqrt(nSubj);
    semV = squeeze(std(smoothdata(projV, "gaussian",10), 0, 1, 'omitnan')) / sqrt(nSubj);

    figure('Color','w','Position',[100 900 1000 360]);
    tiledlayout(1,2,'TileSpacing','compact');
    
    nexttile; hold on; grid on; ylim([0 1]);
    title('Left modes projected onto post', 'Interpreter','none');
    for kk = 1:nK
        fill([t fliplr(t)], [muU(:,kk)'+semU(:,kk)' fliplr(muU(:,kk)'-semU(:,kk)')], ...
            'b', 'FaceAlpha', 0.05, 'EdgeColor', 'none', 'HandleVisibility','off');
        plot(t, muU(:,kk), 'LineWidth', 2);
    end
    xlabel('MR trials'); ylabel('|cosine|');

    nexttile; hold on; grid on; ylim([0 1]);
    title('Right modes projected onto eligibility', 'Interpreter','none');
    for kk = 1:nK
        fill([t fliplr(t)], [muV(:,kk)'+semV(:,kk)' fliplr(muV(:,kk)'-semV(:,kk)')], ...
            'b', 'FaceAlpha', 0.05, 'EdgeColor', 'none', 'HandleVisibility','off');
        plot(t, muV(:,kk), 'LineWidth', 2);
    end
    xlabel('MR trials'); ylabel('|cosine|');

    sgtitle([title_str ' (top modes projected onto teaching directions)'], 'Interpreter','none');
end

end


function plot_error_baseline_then_mr(mse_base_tail, mse_mr, nBaseTail, title_str, nMR)
% mse_base_tail: subj x nBaseTail
% mse_mr       : subj x nMR
mse_base_tail = ensure_subj_by_trials(mse_base_tail);
mse_mr = ensure_subj_by_trials(mse_mr);

Y = [mse_base_tail, mse_mr];
mu = median(smoothdata(Y,'gaussian', 10),1,'omitnan');
if size(Y,1) >= 2
    sem = std(Y,0,1,'omitnan') / sqrt(size(Y,1));
else
    sem = zeros(size(mu));
end

t = 1:numel(mu);

figure('Color','w','Position',[100 100 700 360]); hold on;
fill([t, fliplr(t)], [mu+sem, fliplr(mu-sem)], 'b', 'FaceAlpha',0.15,'EdgeColor','none');
plot(t, mu, 'b', 'LineWidth',2);

xline(nBaseTail + 0.5, 'k-', 'LineWidth', 1.2);

% Optional MR -> washout boundary (if appended)
if size(mu,1) > (nBaseTail + nMR)
    xline(nBaseTail + nMR + 0.5, 'k-', 'LineWidth', 1.2);
end
if size(mu,1) > (nBaseTail + nMR)
    xlabel('Trials (Baseline tail, MR, Washout)');
else
    xlabel('Trials (Baseline tail then MR)');
    ylabel('Tracking MSE');
end
title(title_str, 'Interpreter','none'); grid on;
end


function X = ensure_subj_by_trials(X)
if isempty(X)
    X = nan(1,1);
    return;
end
if isvector(X)
    X = reshape(X, 1, []);
end
if size(X,1) ~= 1 && size(X,2) == 1
    X = X';
end
end

function S = pack_traj_for_plot(target_traj, cursor_traj)
S = struct();
S.target = target_traj;
S.cursor = cursor_traj;
end


function plot_aftereffect_by_freq(AEtot, ~, ~, freqs, title_str)
AEtot = ensure_subj_by_freq(AEtot);

N = size(AEtot,1);

muT = mean(AEtot,1);

if N >= 2
    semT = std(AEtot,0,1) / sqrt(N);
else
    semT = zeros(size(muT));
end

colors = gradual_blue(numel(freqs));

figure('Color','w','Position',[100 100 300 400]);
tiledlayout(1,1,'TileSpacing','compact');

nexttile; hold on; title('Total AE', 'Interpreter','none');
plot_freq_scatter_err(freqs, muT, semT, colors); yline(0,'k--'); grid on; ylim([-1 1]);
xlabel('Hz'); ylabel('\DeltaOG');

sgtitle(title_str, 'Interpreter','none');
end


function X = ensure_subj_by_freq(X)
if isempty(X)
    X = nan(1,1);
    return;
end
if isvector(X)
    X = reshape(X, 1, []);
end
end

function plot_early_late_trajectories(trajEarly, trajLate, dt, dur_sec, title_str)
Ne = min(size(trajEarly.target,2), round(dur_sec/dt));
Nl = min(size(trajLate.target,2),  round(dur_sec/dt));

% Scale can be passed through trajEarly or trajLate if needed.
scale_to_cm = 1;
if isfield(trajEarly,'scale_to_cm'); scale_to_cm = trajEarly.scale_to_cm; end
if isfield(trajLate,'scale_to_cm');  scale_to_cm = trajLate.scale_to_cm;  end

figure('Color','w','Position',[100 100 900 380]);
tiledlayout(1,2,'TileSpacing','compact');

% ==================== EARLY ====================
nexttile; hold on; axis equal; grid on;
title(sprintf('Early MR (first %.2f s)', dur_sec), 'Interpreter','none');

target = trajEarly.target(:,1:Ne);
cursor = trajEarly.cursor(:,1:Ne);

idx = 1:min(200, size(target,2));
kn  = 2;

x = target(1,idx) * scale_to_cm;
y = target(2,idx) * scale_to_cm;
n = numel(x);

sz = linspace(20, 100, numel(1:kn:n));

% Target: light gray -> black
col = zeros(n,1,3);
col(:,1,1) = linspace(0.8, 0, n)';
col(:,1,2) = linspace(0.8, 0, n)';
col(:,1,3) = linspace(0.8, 0, n)';
scatter(x(1:kn:end), y(1:kn:end), sz, squeeze(col(1:kn:end,1,:)), ...
    'filled', 'MarkerFaceAlpha', 0.6);

% Cursor: white -> red
x2 = cursor(1,idx) * scale_to_cm;
y2 = cursor(2,idx) * scale_to_cm;

col2 = zeros(n,1,3);
col2(:,1,1) = linspace(1,   0.6, n)';
col2(:,1,2) = linspace(0.6, 0,   n)';
col2(:,1,3) = linspace(0.6, 0,   n)';
scatter(x2(1:kn:end), y2(1:kn:end), sz, squeeze(col2(1:kn:end,1,:)), ...
    'filled', 'MarkerFaceAlpha', 0.8);

xlabel('x'); ylabel('y');

% Clean legend (dummy handles)
hT = scatter(nan,nan,60,[0 0 0],'filled','MarkerFaceAlpha',0.6);
hC = scatter(nan,nan,60,[1 0 0],'filled','MarkerFaceAlpha',0.8);
legend([hT hC], {'Target','Cursor'}, 'Location','best');

% ==================== LATE ====================
nexttile; hold on; axis equal; grid on;
title(sprintf('Late MR (first %.2f s)', dur_sec), 'Interpreter','none');

target = trajLate.target(:,1:Nl);
cursor = trajLate.cursor(:,1:Nl);

idx = 1:min(200, size(target,2));

x = target(1,idx) * scale_to_cm;
y = target(2,idx) * scale_to_cm;
n = numel(x);

sz = linspace(20, 100, numel(1:kn:n));

% Target: light gray -> black
col = zeros(n,1,3);
col(:,1,1) = linspace(0.8, 0, n)';
col(:,1,2) = linspace(0.8, 0, n)';
col(:,1,3) = linspace(0.8, 0, n)';
scatter(x(1:kn:end), y(1:kn:end), sz, squeeze(col(1:kn:end,1,:)), ...
    'filled', 'MarkerFaceAlpha', 0.6);

% Cursor: white -> red
x2 = cursor(1,idx) * scale_to_cm;
y2 = cursor(2,idx) * scale_to_cm;

col2 = zeros(n,1,3);
col2(:,1,1) = linspace(1,   0.6, n)';
col2(:,1,2) = linspace(0.6, 0,   n)';
col2(:,1,3) = linspace(0.6, 0,   n)';
scatter(x2(1:kn:end), y2(1:kn:end), sz, squeeze(col2(1:kn:end,1,:)), ...
    'filled', 'MarkerFaceAlpha', 0.8);

xlabel('x'); ylabel('y');

hT2 = scatter(nan,nan,60,[0 0 0],'filled','MarkerFaceAlpha',0.6);
hC2 = scatter(nan,nan,60,[1 0 0],'filled','MarkerFaceAlpha',0.8);
legend([hT2 hC2], {'Target','Cursor'}, 'Location','best');

sgtitle(title_str, 'Interpreter','none');
end


function plot_freq_scatter_err(freqs, mu, sem, colors)
plot(freqs, mu, 'k-', 'LineWidth', 1.2, 'HandleVisibility','off');

for k = 1:numel(freqs)
    if any(sem)  % avoids warnings when N=1
        errorbar(freqs(k), mu(k), sem(k), 'k', 'LineStyle','none', 'LineWidth', 1, 'HandleVisibility','off');
    end
    scatter(freqs(k), mu(k), 40, colors(k,:), 'filled', 'MarkerEdgeColor','k');
end
xlim([min(freqs)-0.05 max(freqs)+0.05]);
end

function plot_total_with_baseline_tail(pop_OG_total_with_base, freqs, nBaseTail, nMR, title_str)
% subj x (nBaseTail+nMR+nWash) x freq

if nargin < 4 || isempty(nMR)
    nMR = size(pop_OG_total_with_base,2) - nBaseTail;
end
N = size(pop_OG_total_with_base, 1);
mu  = squeeze(mean(pop_OG_total_with_base, 1));                 % time x freq
sem = squeeze(std(pop_OG_total_with_base, 0, 1)) / sqrt(max(1,N));

colors = gradual_blue(numel(freqs));
t = 1:size(mu,1);

figure('Color','w','Position',[100 100 700 380]); hold on;
for k = 1:numel(freqs)
    plot_shaded(t, mu(:,k), sem(:,k), colors(k,:));
end

xline(nBaseTail + 0.5, 'k-', 'LineWidth', 1.2);

% Optional MR -> washout boundary (if appended)
if size(mu,1) > (nBaseTail + nMR)
    xline(nBaseTail + nMR + 0.5, 'k-', 'LineWidth', 1.2);
end
yline(0,'k--');
yline(-1,'k:');

ylim([-1 1]); xlim([1 max(t)]);
if size(mu,1) > (nBaseTail + nMR)
    xlabel('Trials (Baseline tail, MR, Washout)');
else
    xlabel('Trials (Baseline tail then MR)');
end
ylabel('Total Orthogonal Gain');
title(title_str, 'Interpreter','none');
grid on;

leg_strs = arrayfun(@(f) sprintf('%.2f Hz', f), freqs, 'UniformOutput', false);
legend(leg_strs, 'Location','southeast', 'FontSize', 8);
end

function mse = tracking_mse_delayed(target_traj, cursor_disp_traj, valid_sample_mask, d_vis)
% Compare target and displayed cursor at the visual-delay-aligned time points.

T = size(target_traj,2);
Teff = max(1, T - d_vis);

% Build aligned error sequence at display-time indices
e2 = zeros(1, Teff);
m  = ones(1, Teff);

for t = 1:Teff
    td = t + d_vis;
    err = target_traj(:,td) - cursor_disp_traj(:,td);
    e2(t) = sum(err.^2);

    if ~isempty(valid_sample_mask)
        m(t) = valid_sample_mask(td);
    end
end

den = max(1, sum(m));
mse = sum(e2 .* m) / den;
end
