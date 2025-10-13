%% Session 2 Part 1
% In This script and section of the practical you will learn how to go from
% a simple reinforcement learning to combining different decision relevant
% information and simulating a whole agent to do our task for us. 


%% Load files and  settings
% these commands deletes everything from Matlab's memory and closes all open figures
clear 
close all

% #1) Set the path by modifying the line below to where you have stored the
% scripts for this practical:
MatlabCodePath   = '/Users/nomikikoutsoubari/Documents/GitHub/RL_Oxford_Nomi_notes-answers/computational-models-course-master/Session2';
simulatedDataPath='/Users/nomikikoutsoubari/Documents/GitHub/RL_Oxford_Nomi_notes-answers/computational-models-course-master/Session2/simulatedData'; 
% the following line tells Matlab to look for scripts and data in this
% folder and the folders within this path
addpath(genpath(MatlabCodePath)) 

% This line loads an example schedule we will use for this session
load('stable_first_aversive_schedule.mat','trialVariables');whos % Loads only the variable called trialVariables from the file.
fieldnames(trialVariables) % NK
T = struct2table(trialVariables); % NK for me to understand
open T


% From this loaded information, we now know the reward outcomes (i.e.
% getting a reward or not on each trial). This is what the the reinforcement
% learner is trying to predict. More specifically in our case the reinforcement 
% learner is trying to learn the true probability underlying the outcome sequence. 
% We know this "generative" Reward probability (trueProbability) the learner is trying 
% to approximate as well as the reward magnitudes for each option (See lines below)
opt1Rewarded = trialVariables.opt1Rewarded;             % whether (on each trial), there was an outcome behind option 1 (='1') or behind option 2 (='0')
trueProbabilityOpt1 = trialVariables.trueProbability;   % the probability that was used to determine the outcomes when the schedule was made
magOpt1  = trialVariables.magOpt1;                      % the magnitude of option 1 shown at the time of choice
magOpt2  = trialVariables.magOpt2;
numtrials= length(opt1Rewarded);                        % number of trials in the experiment

%% Building a simple reinforcement learner

% - Simulates how an agent learns reward probabilities from feedback.
% - Pre-allocation improves efficiency and ensures correct dimensions.

% Before we can run a reinforcment learner ,we need to define a learning rate 
% with which the model learns. Later in the session today, we will look at 
% what happens if you change the learning rate, but for now pick a
% relatively low one between 0.05 and .3
alpha = 0.05; % default 0.05

% Pre-define the variables that we want to compute:
% In Matlab it is useful to define how many entries variables will have
% before e.g. filling them using a 'for loop' (see below). The reason for
% this is that makes the code faster, but more importantly, it also allows
% you to check whether you have coded everything correctly by checking that
% each entry has gotten filled and that the size of the variables after a
% loop is the same as before

% Creates two column vectors (each of length numtrials).Fills them
% initially with NaN; ou'll later fill each entry inside a loop that runs through trials.
probOpt1 = nan(numtrials,1); % Predictions about whether there will be a reward if option 1 is selected
delta            = nan(numtrials,1);   % Prediction errors on each trial


% The learning model will now applying the two learning equations, i.e.
% compute the prediction error ([1] in handout) and update the predictions
% for the next trial ([2] in handout) for each trial of the experiment.
% This is done using a 'for loop' that counts from the first trial until
% the last trial (i.e. 'numtrials')

% On the first trial, we assume that the model thinks that the reward could
% be behind either option. In other words, that the probability of getting
% a reward if choosing option 1 is 50%
probOpt1(1) = 0.5;

for t = 1:numtrials-1    % The for-loop is one shorter than the experiment, because we compute the prediction for the next trial
    delta(t) = opt1Rewarded(t)-probOpt1(t);
    probOpt1(t+1) = probOpt1(t) + alpha*delta(t);
end

% As we know that that for every trial only one option is rewarded, we
% also know that the reward expectation for option 1 needs to be the
% opposite than for option two and both need to add up to one. 
probOpt2=1-probOpt1;
% This is only the case in our experiment to make learning simpler, as it would
% theoretically be possible for a learner to learn about two options independently
% holding two separate expectations in mind, if option probabilities were indeed 
% independent from each other. 


%% Looking at the predictions from our reinforcement learner (PLOT predictions)

% What:
    %Plots:
%True probabilities and model predictions over trials.
%Reward magnitudes for both options over trials.
% Why:
%Visualizes how well the learner tracks the underlying reward probabilities.
%Helps understand the effect of alpha and the learning process.

% ------
% Let's plot the predictions of the reinforcement learner over the course
% of the experiment 
% this command opens a new figure in Matlab:
figure('color','w');subplot(2,1,1);
% First we plot the true probability that the learner is trying to learn:
plot(trueProbabilityOpt1,'k') % note: the 'k' tells matlab to plot in black, 'b' would be blue, 'r' red and so on. 
% As always if you want to know more about a specific matlab function use
% the matlab help, it generally lists all possible usage of a function
hold on % this command tells Matlab to add more lines to the same graph (rather than erase the previous plot)
% let's add the true outcomes as '*' symbols
plot(opt1Rewarded,'k*'); % note: the 'k*' tells matlab to plot black '*' symbols
% let's add the predictions from the model
plot(probOpt1,'b'); % note: the 'b' tells matlab to plot in blue
% Let's add labels to the axes, a title and figure legends
xlabel('Trials')
ylabel('Reward probability')
title(['Reinforcement learner with learning rate ' num2str(alpha)]) 
% note for the commands used here: 1) the command 'num2str' allows Matlab to
% plot the number inside learning rate as a string. 2) the square brackets
% ([]) mean that matlab should concatenate the title using the text inside
% the quotation marks ('') and the number inside alpha)
legend('True Probability','Outcomes opt1','Predictions'); % note: the legends are entered in the order in which the variables were added to the plot
ylim([-0.1 1.1]) % this tells Matlab to extend the y-axes from -0.1 to 1.1. We are doing this because the '*'s for outcomes are otherwise on the axes
set(gca,'Fontsize',16);
set(gcf, 'Color', 'w');      % sets figure background NK
set(gca, 'Color', 'w');      % sets axes background NK

subplot(2,1,2);
plot(magOpt1,'g');hold on;plot(magOpt2,'b')
xlabel('Trials')
ylabel('Magnitude')
title(['Magnitudes throughout the task']) 
legend({'Magnitude of option 1';'Magnitude of option 2'}) 
set(gca,'Fontsize',16);
% set(gcf, 'Color', 'w');      % sets figure background NK
% set(gca, 'Color', 'w');      % sets axes background NK
set(findall(gcf,'Type','Legend'), 'Color', 'w', 'Box', 'off', 'TextColor', 'k'); % NK change to transparent legends


% Possible Question Point about RF learners. Etc. 1) What happens if Alpha
% is changed. 2) What happens if alpha can change freely from trial to
% trial?  3) How could you learn about multiple things? What might be a problem if you dont see an option outcome for an extended period of time
% 4) What happens with forgetting?



%NK: 
% 1)
   % Low alpha → smooth blue line in the prediction plot (not so smooth for the volatile block)
   % High alpha → jagged blue line that tracks outcomes more closely(smoother for volatile block)

% 2)
  % dynamic or adaptive learning rate: Instead of a fixed a, the learner
  % adjusts: at = f(uncertainty at trial t)
  % how much it updates based on uncertainty or volatility in the environment. 
  % The learner can approximate Bayesian optimal learning (Behrens et al., 2007)
  % Learns efficiently across both stable and volatile blocks
  % Needs more computation/complexity
  % Harder to implement psychologically in simple models

% 3)
  % If there are two or more independent options, you could hold separate predictions for each
  % Each updated independently with its own delta
  % Problem if an option is not observed for many trials:
        % Its prediction does not get updated, so it becomes stale
        % The learner may overestimate or underestimate its value when finally chosen
        % This is why some models include forgetting or decay for unobserved options
        % in the current model: probOpt2 = 1 - probOpt1; learning about one
        % automatically tells you about the other - not independent

% 4)
  % Forgetting can be implemented by decaying past predictions toward a baseline (e.g., 0.5) over time:
  % λ = forgetting rate (0 = no forgetting, 1 = total reset)
    % Effect:
    % Older outcomes have less influence
    % Can help in volatile environments (don't overweight old info)
    % But in stable environments, forgetting may make learning less accurate, as you lose long-term memory
    % Example:
    % No forgetting → predictions converge smoothly to true probability
    % With forgetting → predictions fluctuate more, don't reach exact probability

% Question	                    Effect / Concept
% Alpha change	                High alpha → fast, sensitive learning; Low alpha → slow, stable learning
% Alpha varies trial-to-trial	    Learner adapts to volatility → Bayesian-like behavior
% Learning multiple options	    Need separate predictions; problem if an option not seen → stale predictions
% orgetting	Older outcomes      decay → faster adaptation in volatile environments, but less stable in stable environments

%% Integration of estimated/learned reward probability with shown magnitudes (UTILITIES)

% What:
    % Computes utility as probability × magnitude for each option.
    % Calculates a decision variable as utility1 - utility2.
    % Applies softmax to transform utilities into choice probabilities.
    % Plots the softmax curve and trial-by-trial probabilities.
% Why:
    % Combines learned probabilities with task rewards to simulate decisions.
    % Softmax converts values into realistic stochastic choice probabilities.

% ------
% Now that we have a first estimate of how people might be
% learning/infering the reward probability we used to generate the
% schedule, we need to find a way to combine this information with the
% presented reward magnitudes in order to simulate behaviours. 


% There are many ways how you can combine pieces of information, but for
% now we will go with the normative/ideal/optimal way of combining reward
% probabilities with magnitudes [This could come in handy if you ever visit Las Vegas!]. 
% Simply multiply the magnitude of each option with its probability to
% create another vector/variable called utility1 and utility 2

keyboard;
 % you want is element-wise multiplication, which uses .*
utility1 = probOpt1 .* magOpt1;   % Option 1 utility  % Tip: use   magOpt1 probOpt1
utility2 = probOpt2 .* magOpt2;   % Option 2 utility  % Tip: use   magOpt2 probOpt2


% Now combute the decision variable by combining/comparing the two options
% utilities for every trial. For our script compute the DecisionVariable in
% favour of option 1 choices.
keyboard;
DecisionVariable = utility1 - utility2; % Tip: use utility1 and utility2
% If utility1 > utility2, the decision variable is positive, meaning the model favors option 1.
% If utility1 < utility2, the decision variable is negative, meaning the model favors option 2.
% If they are equal, the decision variable is 0 → the model is indifferent.


SoftmaxLineX=(min(DecisionVariable):max(DecisionVariable)); % creates a linearly spaced vector from he min to  max value of DecisionVariable in steps of 1 (default in MATLAB)
% inverseT = Small values → high randomness (choices are more uniform)
% inverseT = Large values → more deterministic choices (the higher utility is chosen almost always)
inverseT=.2;   % default here = .2 (fairly stochastic) pick an inverse Temperature, i.e. degrees of stochasticy or randomness

% Softmax function for two options (utility 1 and utility 2)
ChoiceProbability1(:,1)=(exp(utility1.*inverseT))./(exp(utility1.*inverseT) + exp(utility2.*inverseT));
ChoiceProbability2(:,1)=1-ChoiceProbability1(:,1); % Since there are only two options, the probability of choosing option 2 is just the complement of option 1.
SoftmaxLineY(:,1)=1./(exp(SoftmaxLineX.*-inverseT)+1); % related to the softmax / logistic function
% SoftmaxLineX is basically the x-axis for your softmax curve. It is used instead of utility1 - utility2, 
% which is common for plotting a smooth softmax curve across a range of decision variable differences.
% ChoiceProbability1 stores the probability of choosing option 1 for each trial or value.



figure('color',[1 1 1],'name','Softmax with choice probability of all trials');
plot(repmat(SoftmaxLineX',1,size(SoftmaxLineY,2)),SoftmaxLineY,'--k','Linewidth',1);hold on;
plot(DecisionVariable,squeeze(ChoiceProbability1),'x')
xlabel('subjective Utility Difference (A-B)');ylabel('Probability of Choosing Option A');
set(gca,'Fontsize',16);
legend({'Softmax function';'Single Trial probability'});
set(gcf, 'Color', 'w');      % sets figure background NK
set(gca, 'Color', 'w');      % sets axes background NK

% X-axis (SoftmaxLineX): the decision variable
       % Often the difference in utility between the two options:u1-u2
       % Negative values → option 2 is better
       % Zero → both options are equal
       % Positive → option 1 is better

% Y-axis (SoftmaxLineY): the probability of choosing option 1
       % Values range from 0 → 1
       % Shows how likely the model is to choose option 1 as the decision variable changes

% How it looks
       % S-shaped curve (sigmoid):
        % At large negative DV → probability of choosing option 1 ≈ 0
        % At DV = 0 → probability ≈ 0.5 (random choice)
        % At large positive DV → probability ≈ 1

    % Steepness depends on inverse temperature (inverseT):
    % High inverseT → steep, almost deterministic
    % Low inverseT → shallow, more stochastic

%% Generating actual choices from choice probabilities (OBJECTIVE UTILITIES)

    % this part is simulating many agents with the same inverseT/learning rate and without trial-by-trial learning variability. 
    % The variability between agents only comes from the random numbers used to turn probabilities into discrete choices, not from differences in α or β.
 
    % What:
        % Simulates NrSimAgents agents.
        % Generates random numbers to turn choice probabilities into actual choices (Chosen1, ChosenB).
        % Plots binned probabilities of choosing option A based on decision variable.
    % Why:
        % Converts probabilities into discrete simulated behavior.
        % Tests if the model behaves sensibly according to the objective utilities.

% ------
% To simulate what an actual participant would do, you now need to turn those probabilities into discrete choices (A or B)

% Now that we have choice probabilities we are almost there! 
NrSimAgents=80; % simulate XNr of sim participants
for isub=1:NrSimAgents    % Lets make a for-loop for the number of agents we want to simulate
  % Lets generate some choices by first generating some random numbers between 0 and 1
RandNrs=rand(length(ChoiceProbability1),1); % rand(n,1) generates a column vector of random numbers between 0 and 1 — one random number per trial.

% Then we compare those numbers with the Choice Probability of A to
% determine whether A was "choosen" by the simulated agent in that specific trial
Chosen1=RandNrs<=ChoiceProbability1; % this converts probabilities into actual choices.
% If the randomnumber is below or equal the A Choice probability then A was choosen.
% If A isn't chosen, as it is a binary forced choice experiment, B must have been.

ChosenB=~Chosen1;  % The ~ sign means false, i.e. it gives a true/1 if Chosen A is false/0; If A wasn't chosen (Chosen1 = 0), B must have been chosen

SimulatedChoices1(:,isub)=Chosen1; % Now we just have to save the choices of Option A in the column of the agent in the loop (in a matrix)
end


% !!!!! As we don't actually know the exact subjective utility of our subjects,
% at least not immediately before running any models, we can also plot the
% Softmax against objective utiltiy using our knowledge of the true
% underlying probabilities.
% Before fitting your model to data, you don't yet know what those subjective utilities are (they depend on parameters like learning rate, which you haven't estimated yet).
% So, to visualize how the model behaves in principle, you can use the objective (true) utilities — what's actually true in the task — as a stand-in.
% That's what this code does.

% Objective utility for Option 1
ActualUtilityA=trueProbabilityOpt1.*magOpt1;% the true probability that option 1 gives a reward (from your task design). magOpt1 = the reward magnitude associated with option 1. Their product is the expected value/utility for 1
% Objective expected utility of option B
ActualUtilityB=(1-trueProbabilityOpt1).*magOpt2; % the probability of reward for option B is 1 - trueProbabilityOpt1

DecisionVariableObjective=ActualUtilityA-ActualUtilityB; %  compute the difference in objective utility between the two options
% This is the decision variable that the softmax function will later use as input to calculate the probability of choosing A
% You're creating a "theoretical" or "ideal observer" version of your model — one that knows the true underlying structure of the task (true reward probabilities × magnitudes).

% Now we can plot whether the choices we generated roughly follow the
% objective utility (i.e. the values that participants perceive or estimate internally) 
% Transforms trial-by-trial simulated data into a summary plot that shows how choice behavior depends on the decision variable (difference in objective utilities).
% The goal here is to check whether the simulated agents behave sensibly — i.e., do they choose option A more often when it's objectively better?
% by binning A choices by the the objective DecisionVariable. [Reminder: A perfect agent should have a step function at 0]!!-->
% At decision variable = 0 (i.e., equal utilities), the agent suddenly switches from always choosing B to always choosing A.
BinBorders=-80:20:80; % Lets go with -80 to 80 in steps of 20. hese bins divide the range of possible decision variable (DV) values into chunks of width 20.
BinCentre=mean([BinBorders(1:end-1)' BinBorders(2:end)'],2); % Lets compute the centre of the bin so that we know where to plot.
for B=1:(length(BinBorders)-1)  % Loop over all Bins (B indexes bins).
    % Find what samples/trials are included in the current bin using larger and
    % smaller as 
    CurrentIndex=repmat((DecisionVariableObjective>=BinBorders(B)) & (DecisionVariableObjective<BinBorders(B+1)),1,size(SimulatedChoices1,2)); 
    
    % This gives you one mean per bin per agent.
    for a=1:size(SimulatedChoices1,2) % for every simulated agent.
        BinValue(B,a)=mean(SimulatedChoices1(CurrentIndex(:,a),a)); % Compute the average choices for the current bin
            % SimulatedChoices1(:,a) = the vector of that agent's A-choices across all trials (1 = chose A, 0 = chose B).
            % CurrentIndex(:,a) = logical vector marking which trials fall into the current bin.
            % SimulatedChoices1(CurrentIndex(:,a), a) = subset of choices for that agent within that bin.
            % mean(...) = the proportion of A choices in that bin (since mean of 0/1 = proportion of 1s).
    end
end
% Now plot the binned values at the currect locations in the middle of the bin boarders
figure('color',[1 1 1],'name','Actual choices homogenous simulated agent, Binned');
plot(repmat(BinCentre,1,size(BinValue,2)),BinValue,'.-');hold on;
errorbar(BinCentre,mean(BinValue,2),std(BinValue,0,2)./(size(BinValue,2).*.5),'k','Linewidth',1);
xlabel('objective Utility Difference (A-B)');ylabel('Probability of Choosing Option A');
set(gca,'Fontsize',16);
% "When the utility of A was this much higher (or lower) than B, how often did the agent choose A?"


%% Simulation with a distribution of learning rate and inverse temperature (does this mean it is different for each participant?)
 % SUBJECTIVE UTILITIES
 % Computes objective decision variable and plots binned choices.

% What:
    % Samples individual learning rates (alpha) and inverse temperatures (beta) from distributions.
    % Creates a factorial design with stable/volatile schedules and aversive/appetitive conditions.
    % Plots distributions of simulated parameters.
% Why:
    % Simulates inter-individual variability, making simulations more realistic.
    % Useful for testing model recovery: can the fitting procedure recover true parameters?
% How:
    % here the learning rate (α) and inverse temperature (β) are sampled from distributions and therefore different for each simulated agent.

%---------
% Now that we have simulated many agents with exactly the same combination
% of Learning rate and temperature, we want to make sure that our analyses
% later are also sensitive when there is a distribution of both. In other
% words !!we are interested in later seeing the fitting recover both the
% average learning rates and temperatures !!for the agents as well as their
% variability around the group mean. This will ensure that our estimate of
% average model parameter is valid as well as any individual difference
% analysis we might want to do. 
% 
% To show you later how it looks when there are insufficient samples for recovering
% the true parameters from a simulation, we will also simulate a situation with far too few (or uninformative) trials.

Stable1stAversive  =load('stable_first_aversive_schedule.mat','trialVariables');
Stable1stNeutral   =load('stable_first_neutral_schedule.mat','trialVariables');
Volatile1stAversive=load('volatile_first_aversive_schedule.mat','trialVariables');
Volatile1stNeutral =load('volatile_first_neutral_schedule.mat','trialVariables');

% factorial design: 2 (Order: stable-first / volatile-first) × 2 (Valence: aversive / appetitive)
% 80 agents in these conditions:
% "stable" vs. "volatile" reward schedules, and
% "appetitive" (reward-based) vs. "aversive" (loss-avoidance) conditions.
% "Of the 80 samples we have 20 in each of the four schedule categories."
    % Agents 1–40 experience a stable environment first (then volatile later).
    % Agents 41–80 experience the volatile environment first (then stable later).
    % So "stable first" vs. "volatile first" defines a between-subjects factor (or block order).
        % Valence:
        % first 20 agents in each order group (stable-first and volatile-first) are aversive.
        % next 20 agents in each order group are appetitive.

% Of the 80 Samples we have 20 in each of the for schedule categories. For
% simplicity of structure, lets say the first 40 are stable first and the
% first 20 of stable and volatile first are in the aversive condition.

Stable1st =[true(40,1);false(40,1)];% creates a logical vector (80×1) marking which agents have the "stable-first" schedule. ";" concatenates vertically
AversiveCondition=repmat([true(20,1);false(20,1)],2,1); % This just makes a string of 20 true, 20 false for Aversive condition and repeats it 
% (command: repmat(matrix,Rows,Columns)); 
% repmat(..., 2, 1) repeats that 40×1 vector twice vertically (2×). So after repetition, you have 80×1:
    % 1–20	true → Aversive (stable-first group 1)
    % 21–40	false → Appetitive (stable-first group 2)
    % 41–60	true → Aversive (volatile-first group 3)
    % 61–80	false → Appetitive (volatile-first group 4)

% Now we need to draw learning rates and temperatures in order to test
% whether those values can be recovered from simulated agents with our
% schedule. To test whether estimates can recover the true underlying
% parameter values we might want to go with a sensible set of both
% temparatures and learning rates. In This examples lets take learning
% rates in between 0.001 and 1. Inverse Temperature will vary between 0.001
% and 1 for now
minPossibleLR=0.001;   % Define lowest possible Learning rate
maxPossibleLR=1;       % Define largest possible learning rate

% function normrnd(mu, sigma, m, n)
% generates random numbers from a normal (Gaussian) distribution with:
% mu → mean of the distribution
% sigma → standard deviation
% m, n → size (rows × columns) of the output matrix

sampledinverseTemps=log(1+abs(normrnd(0,.7,length(Stable1st),1)));
%| Code                                   | Explanation                                                                                                                                                                                                                                              |
%| -------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
%| `normrnd(0, .7, length(Stable1st), 1)` | Draws a random number from a **normal distribution** with mean 0 and standard deviation 0.7, once for each agent (so 80 total, since `length(Stable1st)=80`).                                                                                            |
%| `abs(...)`                             | Takes the absolute value, because inverse temperature β should not be negative (negative values would flip preferences).                                                                                                                                 |
%| `1 + abs(...)`                         | Adds 1 so we don't take the log of 0 (and to shift all values to be positive).                                                                                                                                                                           |
%| `log(...)`                             | Applies a **log-transform** to make the distribution more right-skewed (lots of small βs, few large βs). This roughly mimics how inverse temperatures are distributed in real human data (many people are somewhat noisy, a few are very deterministic). |
% Result: You get 80 random positive values for inverse temperature (β), mostly between ~0.1 and 1, with a long tail.

sampledLearningrates=normrnd(0.3,.2,length(Stable1st),1);   % Sample learning rates as a normal distribution with mean 0.3 and std of 0.7. 
 %Note, we don't know whether true learning rates will be normally distributed, but in the simulation we want to know whether we could recover
 %individual subject variance and for this normal distribution around mean
 %is a good working assumption

% Make sure Learning rates are bound between 0.001 and 1
% Any α ≤ 0 is replaced by 0.001
% Any α > 1 is replaced by 1
sampledLearningrates(sampledLearningrates<=0)=minPossibleLR;   
sampledLearningrates(sampledLearningrates>1) =maxPossibleLR;

% Now lets plot our simulated Learning rates and inverse Temperatures
figure('color',[1 1 1],'name','Simulated parameter values');
subplot(2,2,1);bar(mean(sampledLearningrates));hold on;
errorbar(mean(sampledLearningrates),std(sampledLearningrates)./(length(sampledLearningrates)^.5),'k','Linewidth',2);
plot(repmat(1,length(sampledLearningrates),1),sampledLearningrates,'.');
xlim([0.5 1.5]);title('Learning Rates')
set(gca,'Fontsize',16);
ylabel('LR value')

subplot(2,2,2);hist(sampledLearningrates,12);hold on;
title('Learning Rates')
set(gca,'Fontsize',16);
ylabel('Number (Histogram)')

subplot(2,2,3);bar(mean(sampledinverseTemps));hold on;
errorbar(1,mean(sampledinverseTemps),std(sampledinverseTemps)./(length(sampledinverseTemps)^.5),'k','Linewidth',2); 
plot(repmat(1,length(sampledinverseTemps),1),sampledinverseTemps,'.');
xlim([0.5 1.5]);title('inverse T')
ylabel('inverse T value')
set(gca,'Fontsize',16);
subplot(2,2,4);hist(sampledinverseTemps,12);hold on;
title('inverse T')
set(gca,'Fontsize',16);
ylabel('Number (Histogram)')

%% Core of simulation loop 

%For 80 agents

    % This code creates artificial participants (agents) who:       
    % - Experience a sequence of trials (like your real participants would),
    % - Learn from feedback using their own learning rate (α),
    % - Make choices based on softmax decision noise (β = inverse temperature).
    % - By simulating 80 such agents, you can later test:   
      % "If real humans behaved like this, could my model recover their parameters?"


% (Hopefully) good simulation
NrOfAgents=length(sampledinverseTemps) ; % length(sampledLearningrates) would also give 80.

%Each variable stores a key internal variable for every trial × every agent.
probOpt1    = nan(numtrials,NrOfAgents);  
probOpt2    = nan(numtrials,NrOfAgents);  
delta               = nan(numtrials,NrOfAgents); 
utility1          = nan(numtrials,NrOfAgents); 
utility2          = nan(numtrials,NrOfAgents); 
ChoiceProbability1= nan(numtrials,NrOfAgents); 
ChoiceProbability2= nan(numtrials,NrOfAgents); 
SimulatedChoices1 = nan(numtrials,NrOfAgents); 
 
for isub=1:NrOfAgents
    % First load the appropriate schedule for a simulated agent using if
    % statements.     
    if Stable1st(isub) && AversiveCondition(isub)
        opt1Rewarded(:,isub)       =Stable1stAversive.trialVariables.opt1Rewarded;
        magOpt1(:,isub)            =Stable1stAversive.trialVariables.magOpt1;
        magOpt2(:,isub)            =Stable1stAversive.trialVariables.magOpt2;
        trueProbabilityOpt1(:,isub)=Stable1stAversive.trialVariables.trueProbability;
        isStableBlock(:,isub)      =[true(size(trueProbabilityOpt1,1)./2,1);false(size(trueProbabilityOpt1,1)./2,1)];
    elseif ~Stable1st(isub) && AversiveCondition(isub)
        opt1Rewarded(:,isub)       =Volatile1stAversive.trialVariables.opt1Rewarded;
        magOpt1(:,isub)            =Volatile1stAversive.trialVariables.magOpt1;
        magOpt2(:,isub)            =Volatile1stAversive.trialVariables.magOpt2;
        trueProbabilityOpt1(:,isub)=Volatile1stAversive.trialVariables.trueProbability;
        isStableBlock(:,isub)      =[false(size(trueProbabilityOpt1,1)./2,1);true(size(trueProbabilityOpt1,1)./2,1)];
    elseif ~Stable1st(isub) && ~AversiveCondition(isub)
        opt1Rewarded(:,isub)       =Volatile1stNeutral.trialVariables.opt1Rewarded;
        magOpt1(:,isub)            =Volatile1stNeutral.trialVariables.magOpt1;
        magOpt2(:,isub)            =Volatile1stNeutral.trialVariables.magOpt2;
        trueProbabilityOpt1(:,isub)=Volatile1stNeutral.trialVariables.trueProbability;
        isStableBlock(:,isub)      =[false(size(trueProbabilityOpt1,1)./2,1);true(size(trueProbabilityOpt1,1)./2,1)];
    elseif Stable1st(isub) && ~AversiveCondition(isub)
        opt1Rewarded(:,isub)       =Stable1stNeutral.trialVariables.opt1Rewarded;
        magOpt1(:,isub)            =Stable1stNeutral.trialVariables.magOpt1;
        magOpt2(:,isub)            =Stable1stNeutral.trialVariables.magOpt2;
        trueProbabilityOpt1(:,isub)=Stable1stNeutral.trialVariables.trueProbability;
        isStableBlock(:,isub)      =[true(size(trueProbabilityOpt1,1)./2,1);false(size(trueProbabilityOpt1,1)./2,1)];
    end
    
    % isStableBlock:creates a column vector of logical values (true/false) marking which trials belong to a stable block.
        % For stable-first blocks: the first half of trials is stable (true), the second half volatile (false).
        % For volatile-first blocks: the first half volatile (false), the second half stable (true).
        % This ensures each agent has a trial-by-trial vector indicating stability.
            % size(...)/2: divides by 2 to split the trials into first half and second half, 40 and 40
            % true(n,1) and false(n,1) -> true(40,1) → a 40×1 column vector of true. false(40,1) → a 40×1 column vector of false. ";" = concatenation

    % Learn from outcomes with specified learning rate
     
    % Initialize choice proba:
    probOpt1(1,isub) = 0.5; % agent starts with no knowledge
    
    % Update beliefs using a learning rule - Rescorla-Wagner / delta-rule update
    for t = 1:numtrials-1 % he code updates the belief for the next trial; so it looks into previous the trial
        delta(t,isub) = opt1Rewarded(t,isub)-probOpt1(t,isub);
        probOpt1(t+1,isub) = probOpt1(t,isub) + sampledLearningrates(isub)*delta(t,isub);
    end
    probOpt2(:,isub)=1-probOpt1(:,isub); % Option 2's probability is just the opposite of option 1  

    % Compute utilities - Combine Probability and Magnitudes
    utility1(:,isub)=magOpt1(:,isub).*probOpt1(:,isub);
    utility2(:,isub)=magOpt2(:,isub).*probOpt2(:,isub);

    % Generate Choice Probabilities using specified Softmax inverse Temperature
    inverseT=sampledinverseTemps(isub);
    ChoiceProbability1(:,isub)=(exp(utility1(:,isub).*inverseT))./(exp(utility1(:,isub).*inverseT) + exp(utility2(:,isub).*inverseT));
    ChoiceProbability2(:,isub)=1-ChoiceProbability1(:,isub);   

    % Generate Choices again from Choice Probabilities
    RandNrs=rand(length(ChoiceProbability1),1);
    Chosen1=RandNrs<=ChoiceProbability1(:,isub);
    ChosenB=~Chosen1;
    SimulatedChoices1(:,isub)=Chosen1;
end
    % Compute "true" expected utility (objective)
 ActualUtilityA=trueProbabilityOpt1.*magOpt1;
ActualUtilityB=(1-trueProbabilityOpt1).*magOpt2;
DecisionVariableObjective=ActualUtilityA-ActualUtilityB;
    
%% Plot subjective utilive
       

% Now we can plot whether the choices we generated roughly follow the
% objective utility by binning A choices by the the objective Decision
% Variable. [Reminder: A perfect agent should have a step function at 0]
BinBorders=-80:20:80; % Lets go with -80 to 80 in steps of 20.
BinCentre=mean([BinBorders(1:end-1)' BinBorders(2:end)'],2); % Lets compute the centre of the bin so that we know where to plot.
for B=1:(length(BinBorders)-1)  % Loop over all Bins
    % Find what samples are included in the current bin using larger and smaller as 
    for a=1:size(SimulatedChoices1,2)
        CurrentIndex=((DecisionVariableObjective(:,a)>=BinBorders(B)) & (DecisionVariableObjective(:,a)<BinBorders(B+1)));
        BinValue(B,a)=mean(SimulatedChoices1(CurrentIndex,a)); % Compute the average choices for the current bin
    end
end
% Now plot the binned values at the currect locations in the middle of the bin boarders
figure('color',[1 1 1],'name','Actual choices  variable simulated agent, Binned');
plot(repmat(BinCentre,1,size(BinValue,2)),BinValue,'.-');hold on;
errorbar(BinCentre,mean(BinValue,2),std(BinValue,0,2)./(size(BinValue,2).*.5),'k','Linewidth',1);
xlabel('objective Utility Difference (A-B)');ylabel('Probability of Choosing Option A');
set(gca,'Fontsize',16);

% Put Choices,true parameters and Schedule Together to save later
simulatedData.opt1Rewarded        =opt1Rewarded;
simulatedData.magOpt1             =magOpt1;
simulatedData.magOpt2             =magOpt2;
simulatedData.trueProbabilityOpt1 =trueProbabilityOpt1;
simulatedData.Learningrates       =sampledLearningrates;
simulatedData.inverseTs           =sampledinverseTemps;
simulatedData.opt1Chosen          =SimulatedChoices1;
simulatedData.chosenOptionRewarded=SimulatedChoices1==opt1Rewarded;
simulatedData.pointswon           =simulatedData.chosenOptionRewarded.*(SimulatedChoices1.*magOpt1 + ~SimulatedChoices1.*magOpt2);
simulatedData.isStableBlock       =isStableBlock;

%% PLOT with fewer samples (still SUBJECTIVE)

% What:
    % Repeats simulation but only with the first 40 trials per agent.
    % Generates choices, utilities, decision variables, and binned plots.

% Why:
    % Demonstrates the effect of limited trial numbers on simulated behavior and parameter recovery.
    % Useful to understand the importance of sample size in modeling experiments.


% For 40 agents
% Now a simulation with too few samples. We just take the first 40 trials of each schedule
trialsforShort=40;
NrOfAgents=length(sampledinverseTemps) ;

shortprobOpt1    = nan(trialsforShort,NrOfAgents);  
shortprobOpt2    = nan(trialsforShort,NrOfAgents);  
shortdelta               = nan(trialsforShort,NrOfAgents); 
shortutility1          = nan(trialsforShort,NrOfAgents); 
shortutility2          = nan(trialsforShort,NrOfAgents); 
shortChoiceProbability1= nan(trialsforShort,NrOfAgents); 
shortChoiceProbability2= nan(trialsforShort,NrOfAgents); 
shortSimulatedChoices1 = nan(trialsforShort,NrOfAgents); 
 
for isub=1:NrOfAgents
    % First load the appropriate schedule for a simulated agent using if statements.     
    if Stable1st(isub) && AversiveCondition(isub)
        shortopt1Rewarded(:,isub)       =Stable1stAversive.trialVariables.opt1Rewarded(1:trialsforShort);
        shortmagOpt1(:,isub)            =Stable1stAversive.trialVariables.magOpt1(1:trialsforShort);
        shortmagOpt2(:,isub)            =Stable1stAversive.trialVariables.magOpt2(1:trialsforShort);
        shorttrueProbabilityOpt1(:,isub)=Stable1stAversive.trialVariables.trueProbability(1:trialsforShort);
        shortisStableBlock(:,isub)      =true(trialsforShort,1);
    elseif ~Stable1st(isub) && AversiveCondition(isub)
        shortopt1Rewarded(:,isub)       =Volatile1stAversive.trialVariables.opt1Rewarded(1:trialsforShort);
        shortmagOpt1(:,isub)            =Volatile1stAversive.trialVariables.magOpt1(1:trialsforShort);
        shortmagOpt2(:,isub)            =Volatile1stAversive.trialVariables.magOpt2(1:trialsforShort);
        shorttrueProbabilityOpt1(:,isub)=Volatile1stAversive.trialVariables.trueProbability(1:trialsforShort);
        shortisStableBlock(:,isub)      =false(trialsforShort,1);
    elseif ~Stable1st(isub) && ~AversiveCondition(isub)
        shortopt1Rewarded(:,isub)       =Volatile1stNeutral.trialVariables.opt1Rewarded(1:trialsforShort);
        shortmagOpt1(:,isub)            =Volatile1stNeutral.trialVariables.magOpt1(1:trialsforShort);
        shortmagOpt2(:,isub)            =Volatile1stNeutral.trialVariables.magOpt2(1:trialsforShort);
        shorttrueProbabilityOpt1(:,isub)=Volatile1stNeutral.trialVariables.trueProbability(1:trialsforShort);
        shortisStableBlock(:,isub)      =false(trialsforShort,1);
    elseif Stable1st(isub) && ~AversiveCondition(isub)
        shortopt1Rewarded(:,isub)       =Stable1stNeutral.trialVariables.opt1Rewarded(1:trialsforShort);
        shortmagOpt1(:,isub)            =Stable1stNeutral.trialVariables.magOpt1(1:trialsforShort);
        shortmagOpt2(:,isub)            =Stable1stNeutral.trialVariables.magOpt2(1:trialsforShort);
        shorttrueProbabilityOpt1(:,isub)=Stable1stNeutral.trialVariables.trueProbability(1:trialsforShort);
        shortisStableBlock(:,isub)      =true(trialsforShort,1);
    end
    
    % Learn from outcomes with specified learning rate
    shortprobOpt1(1,isub) = 0.5;
    for t = 1:trialsforShort-1
        shortdelta(t,isub) = shortopt1Rewarded(t,isub)-shortprobOpt1(t,isub);
        shortprobOpt1(t+1,isub) = shortprobOpt1(t,isub) + sampledLearningrates(isub)*shortdelta(t,isub);
    end
    shortprobOpt2(:,isub)=1-shortprobOpt1(:,isub);       
    % Combine Probability and Magnitudes
    shortutility1(:,isub)=shortmagOpt1(:,isub).*shortprobOpt1(:,isub);
    shortutility2(:,isub)=shortmagOpt2(:,isub).*shortprobOpt2(:,isub);
    % Generate Choice Probabilities using specified Softmax inverse Temperature
    inverseT=sampledinverseTemps(isub);
    shortChoiceProbability1(:,isub)=(exp(shortutility1(:,isub).*inverseT))./(exp(shortutility1(:,isub).*inverseT) + exp(shortutility2(:,isub).*inverseT));
    shortChoiceProbability2(:,isub)=1-shortChoiceProbability1(:,isub);   
    % Generate Choices again from Choice Probabilities
    RandNrs=rand(size(shortChoiceProbability1,1),1);
    Chosen1=RandNrs<=shortChoiceProbability1(:,isub);
    ChosenB=~Chosen1;
    shortSimulatedChoices1(:,isub)=Chosen1;
end


shortActualUtilityA=shorttrueProbabilityOpt1.*shortmagOpt1;
shortActualUtilityB=(1-shorttrueProbabilityOpt1).*shortmagOpt2;
shortDecisionVariableObjective=shortActualUtilityA-shortActualUtilityB;
BinBorders=-80:80:80; % Because we have so little data we just have two bins here
BinCentre=mean([BinBorders(1:end-1)' BinBorders(2:end)'],2); % Lets compute the centre of the bin so that we know where to plot.
BinValue=[];
for B=1:(length(BinBorders)-1)  % Loop over all Bins
    % Find what samples are included in the current bin using larger and
    % smaller as 
    for a=1:size(SimulatedChoices1,2)
        CurrentIndex=((shortDecisionVariableObjective(:,a)>=BinBorders(B)) & (shortDecisionVariableObjective(:,a)<BinBorders(B+1)));
        BinValue(B,a)=mean(shortSimulatedChoices1(CurrentIndex,a)); % Compute the average choices for the current bin
    end
end
% Now plot the binned values at the currect locations in the middle of the bin boarders
figure('color',[1 1 1],'name','Actual choices variable short simulated agent, Binned');
plot(repmat(BinCentre,1,size(BinValue,2)),BinValue,'.-');hold on;
errorbar(BinCentre,nanmean(BinValue,2),nanstd(BinValue,0,2)./(size(BinValue,2).*.5),'k','Linewidth',1);
xlabel('objective Utility Difference (A-B)');ylabel('Probability of Choosing Option A');
set(gca,'Fontsize',16);
% Put Choices,true parameters and Schedule Together to save later



%% This bit you dont have to look at. It simply plots the aggregate behaviours etc.


ProbabibilityOption1=trueProbabilityOpt1;
ProbabibilityOption2=1-trueProbabilityOpt1;   

ActualUtility1=ProbabibilityOption1.*magOpt1;
ActualUtility2=ProbabibilityOption2.*magOpt2;
ActualUDiff=ActualUtility1-ActualUtility2;


for SubjectIndex=1:size(SimulatedChoices1,2)         % the second dimension, i.e. nr of columns is equal to number of participants.
    OptionA_BetterOrEqualB=(ActualUDiff(:,SubjectIndex)>=0); %Define which option is correct for a subject by whether it has the higher utility
    CorrectChoice(:,SubjectIndex)=SimulatedChoices1(:,SubjectIndex)==OptionA_BetterOrEqualB; % Compare choices of A with whether that was better to compute correctness
    accuracies(SubjectIndex,1)=mean(CorrectChoice(:,SubjectIndex));   % Average all choices of a participant in order compute overall accuracy.   
end

% Plotting stuff
figure('color',[1 1 1],'name','Aggregate Measurements');subplot(2,1,1);hold on;
bar(1,mean(accuracies ));errorbar(1,mean(accuracies ),(std(accuracies )./(size(CorrectChoice,2).^.5)),'.k','Linewidth',4);
plot(ones(length(accuracies ),1),accuracies ,'.');
set(gca,'Fontsize',16);


for SubjectIndex=1:size(SimulatedChoices1,2)         % the second dimension, i.e. nr of columns is equal to number of participants.
    ProbabilityEstimate1(:,SubjectIndex)=probOpt1(:,SubjectIndex);
    ProbabilityEstimate2(:,SubjectIndex)=1-probOpt1(:,SubjectIndex);
    
    SubjUtility1(:,SubjectIndex)=ProbabilityEstimate1(:,SubjectIndex).*magOpt1(:,SubjectIndex) ;
    SubjUtility2(:,SubjectIndex)=ProbabilityEstimate2(:,SubjectIndex).*magOpt2(:,SubjectIndex) ;
    SubjUDiff(:,SubjectIndex)=SubjUtility1(:,SubjectIndex)-SubjUtility2(:,SubjectIndex);
    subjectiveOption1_BetterOrEqual2=SubjUDiff(:,SubjectIndex)>=0;
    SubjectiveCorrectChoice(:,SubjectIndex)=SimulatedChoices1(:,SubjectIndex)== subjectiveOption1_BetterOrEqual2;
    subjectiveaccuracies(SubjectIndex,1)=mean(SubjectiveCorrectChoice(:,SubjectIndex));
end
bar(2,mean(subjectiveaccuracies ));errorbar(2,mean(subjectiveaccuracies ),(std(subjectiveaccuracies )./(size(SubjectiveCorrectChoice ,2).^.5)),'.k','Linewidth',4);
plot(2*ones(length(subjectiveaccuracies ),1),subjectiveaccuracies ,'.'); plot([0.5 2.5],[.5 .5],'k','Linewidth',1)
%set(gca,'XTick',[1 2],'XTickLabel',{'obj. Accuracy';'Subj. Accuracy'},'XTickLabelRotation',45)
% to allow for different matlab version allowing or not allowing rotated labels 
try 
set(gca,'XTick',[1 2],'XTickLabel',{'obj. Accuracy';'Subj. Accuracy'},'XTickLabelRotation',45,'Fontsize',16)
catch
set(gca,'XTick',[1 2],'XTickLabel',{'obj. Accuracy';'Subj. Accuracy'},'Fontsize',16)    
end
ylabel('Probability of being correct');
xlim([0.5 2.5])

%% Win/Stay Lose/Shift

% Step 1: Determine if the previous choice was rewarded
    % opt1Rewarded is the actual outcome for option 1 (1 = rewarded, 0 not rewarded). LOADED in the beginnign
    % SimulatedChoices1 indicates whether the agent chose option 1 (1 = chose, 0 = didn't).
    % ChoiceRewarded is a matrix indicating whether the agent's choice was rewarded on each trial.
ChoiceRewarded=opt1Rewarded==SimulatedChoices1;% 1 if the agent's choice matches the reward status of option 1, and 0 otherwise.
% == operator – In MATLAB, this compares two arrays element-wise and returns 1 (true) if the elements are equal, and 0 (false) if not.

subplot(2,1,2);
% Stay measures whether the agent repeated the same choice as the previous trial.
    % SimulatedChoices1(1:end-1,:) → all trials except the last
    % SimulatedChoices1(2:end,:) → all trials except the first
    % == → checks if choice repeated from previous trial
    % [nan(1,size(SimulatedChoices1,2)); ... ] → adds a NaN for the first trial (no previous trial to compare)
            % 1 → agent stayed with previous choice
            % 0 → agent switched
            % NaN → first trial
Stay=[nan(1,size(SimulatedChoices1,2));SimulatedChoices1(1:end-1,:)==SimulatedChoices1(2:end,:)];

% Whether agent received a reward.
    % 1:end-1 → takes all rows except the last. Because we want to shift the reward information down so that trial 2 can "see" trial 1's reward.
    % nan(1, size(ChoiceRewarded,2)) → a row of NaN with same number of columns (agents)
    % Why? The first trial has no previous trial, so its "last trial reward" is undefined → we use NaN
        % This "shift down by one row and pad with NaN" is a common trick in MATLAB 
        % to create a vector of "previous trial values" aligned with the current trial. 
        % It's used for trial-by-trial analyses, like Win-Stay / Lose-Shift.
   LastTrialReward=[nan(1,size(ChoiceRewarded,2));ChoiceRewarded(1:end-1,:)];

for a=1:size(ChoiceRewarded,2)
    WinStay(a) =mean(Stay(LastTrialReward(:,a)==1,a)); % LastTrialReward(:,a)==1 → selects trials where last trial was rewarded; Stay(...) → takes whether agent stayed after those trials
    LoseStay(a)=mean(Stay(LastTrialReward(:,a)==0,a)); % LoseStay → mean probability of staying after an unrewarded trial (same logic)
end

subplot(2,1,2);hold on;
bar([mean(WinStay ) mean(LoseStay )]);errorbar([mean(WinStay ) mean(LoseStay )],([std(WinStay ) std(LoseStay )]./(length(WinStay).^.5)),'.k','Linewidth',4);
plot([WinStay' LoseStay']','--');
xlim([0.5 2.5])
try
set(gca,'XTick',[1 2],'XTickLabel',{'Win Stay';'Lose Stay'},'XTickLabelRotation',45,'Fontsize',16)
catch
set(gca,'XTick',[1 2],'XTickLabel',{'Win Stay';'Lose Stay'},'Fontsize',16)
end
ylabel('Probability of Staying');



%% Optional Bit. Multiple regressions

NrBack=10;                                          % How many trials to go back in tracking the outcome history.

% Loop over subjects and trials to build past reward history
  % b is the number of trials back you are looking at. b = 1 → t-1 (last trial), b = 2 → t-2 (2 trials ago)
for isub=1:size(opt1Rewarded,2)% loop over subjects/agents
    for b=1:NrBack % loop over how many trials back you want to look
            Pastopt1Rewarded(:,b,isub)=[nan(b,1);opt1Rewarded(1:end-b,isub)]; % shift the rewards down by b trials, 
                                            % nan(b,1) → creates a column vector of b NaNs. Because for the first b trials, there is no trial b steps back.
                                            % opt1Rewarded(1:end-b,isub) →
                                            % takes the reward history of
                                            % subject isub from trial 1 to trial end-b, shifting the reward history down by b trials
                                            % Pastopt1Rewarded(:,b,isub) → 3D matrix:   
                                                % Rows → trials
                                                % Columns → "how many trials back" (t-1, t-2, … t-10)
                                                % 3rd dim → subjects
                                            % Result: Pastopt1Rewarded(trial, b, subject) tells you if trial b trials ago was rewarded.
    end
end

% Label the design matrix
DesignMatrixLabels={'t-1';'t-2';'t-3';'t-4';'t-5';'t-6';'t-7';'t-8';'t-9';'t-10'};

% Fit logistic regression per subject
for isub=1:size(opt1Rewarded,2)
    % Take out?
    y=SimulatedChoices1(:,isub);  %whether the agent chose option 1 (1) or not (0)
    DesignMatrix=Pastopt1Rewarded(:,:,isub);
    %
[betaweights(:,isub),devf,stats{isub}]=glmfit(DesignMatrix,y,'binomial','link','logit'); % logistic regression; smaller deviance (defv) = better fit

%DesignMatrix → matrix of predictors
%Each column is a feature (here: reward history t-1, t-2, … t-10)
% Each row is a trial

%betaweights(t-1) → effect of reward 1 trial back
%betaweights(t-2) → effect of reward 2 trials back, etc.

%Key idea:
%If betaweight(t-1) is positive → if last trial was rewarded, subject more likely to repeat choice ("Win-Stay")
% If negative → more likely to switch ("Win-Shift")

end

% combine betaweights and stats to see which past trials significantly influence choice:
% Positive β → past reward increases probability of staying
for b = 2:size(betaweights,1) % skip intercept
    fprintf('Trial t-%d: beta=%.3f, p=%.3f\n', b-1, betaweights(b,isub), stats{isub}.p(b));
end


% SE
stats{isub}.se
stats{isub}.p
stats{isub}.t
yhat = glmval(betaweights(:,isub), DesignMatrix, 'logit'); % predicted probabilitues; 
% glmval takes the coefficients (betaweights) and applies the logistic function to the linear combination of predictors in DesignMatrix.
% predicted probability that the subject will choose option 1 on that trial, given the past rewards.
% Values close to 1 → model predicts the subject will almost certainly choose option 1.





figure('color',[1 1 1],'name','Results of Multiple regression');
hold on;
plotData=betaweights(2:end,:);
bar(mean(plotData,2));hold on;%plot(plotData,'.','markersize',8);
errorbar(mean(plotData,2),std(plotData,0,2)./size(plotData,2).^.5,'.k');
%plot(plotData,'.','markersize',20)
set(gca,'XTick',1:length(DesignMatrixLabels),'XTickLabel',DesignMatrixLabels)
ylabel('effect size on choices (betaweights, arbitrary units)')
set(gca,'Fontsize',16);

