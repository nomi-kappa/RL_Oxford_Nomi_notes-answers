function probOpt1 = RL_model(opt1Rewarded,alpha,startingProb)

% probOpt1 = RL_model(opt1Rewarded,alpha,startingProb)
% 
% opt1rewarded = vector of outcomes (1 if opt 1 rewarded, 0 if opt 2 rewarded)
% alpha = fixed learning rate, greater than 0, less than/equal to 1
% startingProb = starting probability (defaults to 0.5)
%
% probOpt1 is returned as a vector of how likely option 1 is to be rewarded

%% CHECK THE INPUT ARGUMENTS

% check alpha has been set appropriately
if alpha <= 0
    error('Learning rate (alpha) must be greater than 0');
elseif alpha > 1
    error('Learning rate (alpha) must be less than or equal to 1');
end

% set the starting probability to 0.5 if it hasn't been given
if nargin < 3 % the number of function input arguments (have already set three of them in session1.m- here just for flexibility reasons)
   startingProb = 0.5; 
end

% set the first trial's prediction to be equal to the starting probability
probOpt1(1) = startingProb;

%calculate the number of trials
nTrials = length(opt1Rewarded);

%% STUDENTS - complete this code to finish the reinforcement learning model

for t = 1:nTrials %loop over trials
    delta(t) = opt1Rewarded(t) - probOpt1(t); %%COMPLETE THIS LINE using opt1Rewarded, probOpt1 and equation 1 %%; % prediction error [δt = Ot - pt]
    probOpt1(t+1) = probOpt1(t) + alpha*delta(t);  %%COMPLETE THIS LINE  using probOpt1, delta, alpha and equation 2 %%;   % prediction for next trial [pt+1 = pt + a*δt]
end