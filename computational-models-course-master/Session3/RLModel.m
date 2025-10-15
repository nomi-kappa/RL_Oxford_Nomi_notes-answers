function [e] = RLModel(p,subjN,data,plot_fits)

% In case we ever want to fit two different learning rates 'alpha' and one 
% or two different inverse temperatures 'beta', make the script flexible
% and duplicate if only one was specified
if length(p)==2
    learningRate = [p(1) p(1)];  % learning rate
    inverseT     = [p(2) p(2)];  % inverse temparature
elseif length(p)==3
    learningRate = [p(1) p(2)];  % two learning rates
    inverseT     = [p(3) p(3)];  % one temperature
elseif length(p)==4
    learningRate = [p(1) p(2)];  % two learning rates
    inverseT     = [p(3) p(4)];  % two temperatures
end

% check that parameters are within legal constraints
if any(learningRate<0) || any(inverseT<0) || any(learningRate>1)
    e = 10000; % give back a large error term (i.e. 'bad fit')
    return
end

% Define variables for this subject
opt1Rewarded = data.opt1Rewarded(:,subjN);             % whether (on each trial), there was an outcome behind option 1 (='1') or behind option 2 (='0')
magOpt1  = data.magOpt1(:,subjN);                      % the magnitude of option 1 shown at the time of choice
magOpt2  = data.magOpt2(:,subjN);
opt1chosen = data.opt1Chosen(:,subjN);
trueProbability = data.trueProbability(:,subjN);
isStable = 2-data.isStableBlock(:,subjN);              % turns 1=stable/0=volatile into: 1=stable; 2=volatile
numtrials= size(opt1Rewarded,1);                       % number of trials in the experiment


% The below lines of code were taken from RL section of session 2
% ====================================================================== %
% Pre-define the variables that we want to compute:
% In Matlab it is useful to define how many entries variables will have
% before e.g. filling them using a 'for loop' (see below). The reason for
% this is that makes the code faster, but more importantly, it also allows
% you to check whether you have coded everything correctly by checking that
% each entry has gotten filled and that the size of the variables after a
% loop is the same as before
predictionOpt1 = nan(numtrials,1); % Predictions about whether there will be a reward if option 1 is selected
rpe        = nan(numtrials,1);   % Prediction errors on each trial


% The learning model will now applying the two learning equations, i.e.
% compute the prediction error ([1] in handout) and update the predictions
% for the next trial ([2] in handout) for each trial of the experiment.
% This is done using a 'for loop' that counts from the first trial until
% the last trial (i.e. 'numtrials')

% On the first trial, we assume that the model thinks that the reward could
% be behind either option. In other words, that the probability of getting
% a reward if choosing option 1 is 50%
predictionOpt1(1) = 0.5;

for t = 1:numtrials-1    % The for-loop is one shorter than the experiment, because we compute the prediction for the next trial
    % #2: Complete the equation to compute prediction error on trial t
    rpe(t) = opt1Rewarded(t)-predictionOpt1(t);
    % #3: Complete the equation to compute the new prediction for the next
    % trial (t+1)
    predictionOpt1(t+1) = predictionOpt1(t) + learningRate(isStable(t))*rpe(t);
end

% As we know that that for every trial only one option is rewarded, we
% also know that the reward expectation for option 1 needs to be the
% opposite than for option two and both need to add up to one. 
predictionOpt2=1-predictionOpt1;
% This is only the case in our experiment to make learning simpler, as it would
% theoretically be possible for a learner to learn about two options independently
% holding two separate expectations in mind, if option probabilities were indeed 
% independent from each other. 

% This should be switched off when fitting many participants, but for
% evaluating a single participant, it can be handy to plot the important
% variables
if plot_fits
    figure('color','w','name',['Participant',num2str(subjN)]);
    plot(trueProbability,'k'); hold on;
    plot(data.chosenOptionRewarded(:,subjN),'k*');
    plot((opt1chosen+0.05).*0.9,'r.');
    xlabel('Trials');
    ylabel('Reward probability');
    ylim([-0.1 1.1]);
    plot(predictionOpt1,'b'); % note: the 'b' tells matlab to plot in blue
    set(gcf, 'Color', 'w');      % sets figure background NK
set(gca, 'Color', 'w');      % sets axes background NK

end


% The below lines of code were taken from section on softmax from session 2
% ====================================================================== %

utility1=magOpt1.*predictionOpt1; 
utility2=magOpt2.*predictionOpt2;

% Decision variable
DV = utility2-utility1;

% for all trials, calculate the two choice probabilities, then save the one
% of the chosen option as 'probChoice'
for t=1:numtrials
    ChoiceProbabilityB(t,1)=1/(1+exp(-inverseT(isStable(t))*DV(t)));
    ChoiceProbabilityA(t,1)=1-ChoiceProbabilityB(t,1);

    if opt1chosen(t)==1
        probChoice(t) = ChoiceProbabilityA(t);
    else
        probChoice(t) = ChoiceProbabilityB(t);
    end
end

if plot_fits
    plot(ChoiceProbabilityA,'g.');
    legend('True Probability','Outcomes received','Choice','PredictionOpt1','ChoiceProbOpt1');    
end

% Add up the error terms (->log likelihood) for the trials used for fitting
TrialsusedForFitting = intersect([1:numtrials],[11:100 111:200]);
e= -sum(log(probChoice(TrialsusedForFitting))); %if probChoice==0 then log(0)=-Inf i.e. penalizes being confident at predicting wrong choice


