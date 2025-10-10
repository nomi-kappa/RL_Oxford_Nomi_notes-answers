% Learning rates, and how they affect probability learning
% Block Practical session 1

%% 1. Learning with a fixed schedule
% this sets up the 'schedule', i.e. which option is rewarded on each trial

fixedProb = 0.8;    % this is the true probability that green is rewarded
startingProb = 0.5;  % this defines the model's estimated probabilty on the very first trial
alpha = 0.05;         % this is the model's learning rate; default 0.05
nTrials = 200;       % this is the number of trials
trueProbability = ones(1,nTrials)*fixedProb; %reward probability on each trial; ones(1, nTrials) → creates a row vector of length nTrials (200) filled with 1s.
                        % Each element in the vector is multiplied by
                        % fixedProb (0.8). 1*0.8, 1*0.8...etc.
                        % So the vector becomes [0.8, 0.8, 0.8, ..., 0.8] of length 200.

% REVERSALS switch of probability through a mat file
% overwrites trueProbability with a schedule that reverses
D = load('schedule.mat'); % trials 1–17 → 0.2 (green rewarded only 20% of the time); 18–51 → 0.8 (green rewarded 80% of the time);
                          % 52–68 → back to 0.2; 69–102 → 0.8 again; 103–200 → stabilises at 0.25 (green rewarded 25% of the time for the rest of the experiment).
% view it schedule.mat
whos -file schedule.mat
trueProbability = D.trueProbabilityStored; % To get the actual vector out of D, you use a dot .:
nTrials = length(trueProbability);

% now, we'll simulate whether green was rewarded on every trial
opt1Rewarded(1:nTrials) = rand(1,nTrials) < trueProbability; % Generates a row vector of random numbers (UNIFORM DISTRIBUTION) between 0 and 1, of length nTrials (here, 200).
                                % trueProbability is your vector of true reward probabilities (all 0.8 here).
                                % The < operator compares each random number to the corresponding element in trueProbability.
                                % example outcome: opt1Rewarded = [1 1 0 1 1 0 1 ...]  % 1 = rewarded, 0 = not rewarded
                                % The < operator compares each random number to the corresponding element in trueProbability.
                                % Over many trials, the fraction of rewards
                                % will approximate trueProbability (0.8 here). WHy? 
                                % You generate a random number between 0 and 1. If the number is less than 0.8 → you get a reward.

%plot the schedule on each trial, and the fixed probability of reward
figure(1);clf;hold on;plotIndex = []; % clf clears the figure so it's fresh; hold on allows you to plot multiple things on the same axes.
plotIndex(1) = plot(trueProbability,'k--','LineWidth',2); % trueProbability is your ground truth (e.g. always 0.8). 'k--' → black dashed line.
%'LineWidth',2 → makes the line thicker for visibility.
plotIndex(2) = plot(opt1Rewarded,'g*','MarkerSize',10); % 'g*' → green star markers.
xlim([0 nTrials]); %sets the x axis
xlabel('Trial number'); %labels it
ylim([0 1]); %sets the y axis
ylabel('Green rewarded?'); %labels it
legend(plotIndex, {'True Probability' 'Trial outcomes'});

% EXERCISE A: open up RL_model.m, and complete the last two lines of code
probOpt1 = RL_model(opt1Rewarded,alpha,startingProb); % RL_model is a function (in RL_model.m) with three arguments;  
% probOpt1 is the model’s estimated probability that green is rewarded on each trial.

%once you've successfully coded your RL_model, you can then plot the model's
%estimated probabilities (on top of the plot above)
plotIndex(3) = plot(probOpt1,'r-','LineWidth',2);
legend(plotIndex, {'True Probability' 'Trial outcomes' 'RL model probability'});

% EXERCISE B. Use the code above to try to understand what happens when you
%   vary alpha (and other parameters). What are the advantages of having a 
%   low alpha? What are the advantages of having a high alpha? If you set
%   alpha to 1, how does the model behave? Make changes to the variables at
%   the top of this cell to explore this (see handout for further details).

% EXERCISE C. Load in the 'reversal' schedule used in the experiment. In
%   which part of the schedule does it help to have a higher alpha? In which
%   part of the schedule does it help to have a lower alpha? (see handout for
%   further details).


%% 3. How many trials do we look back into the past with different alphas?
    % The goal of this code is to visualize how much influence past trials
    % have in an exponentially weighted learning rule, depending on the learning rate.
    % α determines how quickly past information is forgotten — i.e., how much weight is given to recent vs. older trials.

alpha = 0.15; % means that 15% of the new information is incorporated each trial. 
              % The remaining 85% (1-a = 0.85) of the belief or estimate is carried over from the past.


% We are computing the effective weight that a trial t in the past contributes to the current estimate after T total trials.
T = 25;
for t = 1:T             
    % EXERCISE D. explain/derive the following equation:
        % the weight(t) formula: comes from the mathematical form of an exponential moving average (EMA) — which is the core of many RL and prediction update rules.
        % The most recent trial (t = T) contributes:(1−α) (T−(T−1)) ⋅α=(1−α) 1 ⋅α=0.85×0.15 → slightly smaller.
        % Two trials before that: (1−α) 2 ⋅α=0.85 2 ×0.15 → even smaller, and so on.
        % So this creates an exponential decay of influence — older trials matter less.
        % T-t 1st trial, 0.15*0.85, 2nd trial 0.15*0.85^2, 5th trial
        % 0.15*0.85^5...etc
    weight(t) = (1-alpha).^(T-t)*alpha; 
end

%plot the weights running back across previous trials
figure;clf;
plot(-(T-1):1:0,weight);
ylim([0 alpha*1.2]);
xlim([-(T-1) 0]);
xlabel('Delay (trials)');
ylabel('Weight');
title(sprintf('alpha = %0.2f',alpha));
