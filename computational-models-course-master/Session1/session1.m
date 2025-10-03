% Learning rates, and how they affect probability learning
% Block Practical session 1

%% 1. Learning with a fixed schedule
% this sets up the 'schedule', i.e. which option is rewarded on each trial

fixedProb = 0.8;    % this is the true probability that green is rewarded
startingProb = 0.5;  % this defines the model's estimated pronbabilty on the very first trial
alpha = 0.05;         % this is the model's learning rate
nTrials = 200;       % this is the number of trials
trueProbability = ones(1,nTrials)*fixedProb; %reward probability on each trial

% now, we'll simulate whether green was rewarded on every trial
opt1Rewarded(1:nTrials) = rand(1,nTrials) < trueProbability;

%plot the schedule on each trial, and the fixed probability of reward
figure(1);clf;hold on;plotIndex = [];
plotIndex(1) = plot(trueProbability,'k--','LineWidth',2); 
plotIndex(2) = plot(opt1Rewarded,'g*','MarkerSize',10);
xlim([0 nTrials]); %sets the x axis
xlabel('Trial number'); %labels it
ylim([0 1]); %sets the y axis
ylabel('Green rewarded?'); %labels it
legend(plotIndex, {'True Probability' 'Trial outcomes'});

% EXERCISE A: open up RL_model.m, and complete the last two lines of code
probOpt1 = RL_model(opt1Rewarded,alpha,startingProb);

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

alpha = 0.15;

T = 25;
for t = 1:T             
    % EXERCISE D. explain/derive the following equation:
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
