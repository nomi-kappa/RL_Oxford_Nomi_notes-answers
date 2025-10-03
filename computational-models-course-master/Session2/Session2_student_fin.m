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
MatlabCodePath   = '/Users/nkolling/Documents/teaching/UndergraduateTeaching/NewVersionAdvancedPractical';
simulatedDataPath='/Users/nkolling/Documents/teaching/UndergraduateTeaching/NewVersionAdvancedPractical/simulatedData'; 
% the following line tells Matlab to look for scripts and data in this
% folder and the folders within this path
addpath(genpath(MatlabCodePath)) 

% This line loads an example schedule we will use for this session
load('stable_first_aversive_schedule.mat','trialVariables');

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
% Before we can run a reinforcment learner ,we need to define a learning rate 
% with which the model learns. Later in the session today, we will look at 
% what happens if you change the learning rate, but for now pick a
% relatively low one between 0.05 and .3
alpha = 0.05;

% Pre-define the variables that we want to compute:
% In Matlab it is useful to define how many entries variables will have
% before e.g. filling them using a 'for loop' (see below). The reason for
% this is that makes the code faster, but more importantly, it also allows
% you to check whether you have coded everything correctly by checking that
% each entry has gotten filled and that the size of the variables after a
% loop is the same as before
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


%% Looking at the predictions from our reinforcement learner
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
legend('True Probability','Outcomes opt1','Predictions') % note: the legends are entered in the order in which the variables were added to the plot
ylim([-0.1 1.1]) % this tells Matlab to extend the y-axes from -0.1 to 1.1. We are doing this because the '*'s for outcomes are otherwise on the axes
set(gca,'Fontsize',16);

subplot(2,1,2);
plot(magOpt1,'g');hold on;plot(magOpt2,'b')
xlabel('Trials')
ylabel('Magnitude')
title(['Magnitudes throughout the task']) 
legend({'Magnitude of option 1';'Magnitude of option 2'}) 
set(gca,'Fontsize',16);

% Possible Question Point about RF learners. Etc. 1) What happens if Alpha
% is changed. 2) What happens if alpha can change freely from trial to
% trial?  3) How could you learn about multiple things? What might be a problem if you dont see an option outcome for an extended period of time
% 4) What happens with forgetting?


%% Integration of estimated/learned reward probability with shown magnitudes
% Now that we have a first estimate of how people might be
% learning/infering the reward probability we used to generate the
% schedule, we need to find a way to combine this information with the
% presented reward magnitudes in order to simulate behaviours. 


% There are many ways how you can combine pieces of information, but for
% now we will go with the normative/ideal/optimal way of combining reward
% probabilities with magnitudes [This could come in handy if you ever visit Las Vegas!]. 
% Simply multiply the magnitude of each option with its probability to
% create another vector/variable called utility1 and utility 2

keyboard
utility1= ;  % Tip: use   magOpt1 probOpt1
utility2= ;  % Tip: use   magOpt1 probOpt1
%

% Now combute the decision variable by combining/comparing the two options
% utilities for every trial. For our script compute the DecisionVariable in
% favour of option 1 choices.
keyboard;
DecisionVariable=; % Tip: use utility1 and utility2
%

SoftmaxLineX=(min(DecisionVariable):max(DecisionVariable));

inverseT=.2;   % pick an inverse Temperature, i.e. degrees of stochasticy or randomness
ChoiceProbability1(:,1)=(exp(utility1.*inverseT))./(exp(utility1.*inverseT) + exp(utility2.*inverseT));
ChoiceProbability2(:,1)=1-ChoiceProbability1(:,1);
SoftmaxLineY(:,1)=1./(exp(SoftmaxLineX.*-inverseT)+1);


figure('color',[1 1 1],'name','Softmax with choice probability of all trials');
plot(repmat(SoftmaxLineX',1,size(SoftmaxLineY,2)),SoftmaxLineY,'--k','Linewidth',1);hold on;
plot(DecisionVariable,squeeze(ChoiceProbability1),'x')
xlabel('subjective Utility Difference (A-B)');ylabel('Probability of Choosing Option A');
set(gca,'Fontsize',16);
legend({'Softmax function';'Single Trial probability'});

%% Generating actual choices from choice probabilities
% Now that we have choice probabilities we are almost there! 
NrSimAgents=80;
for isub=1:NrSimAgents    % Lets make a for-loop for the number of agents we want to simulate
  % Lets generate some choices by first generating some random numbers between 0 and 1
RandNrs=rand(length(ChoiceProbability1),1);
% Then we compare those numbers with the Choice Probability of A to
% determine whether A was "choosen" by the simulated agent in that specific trial
Chosen1=RandNrs<=ChoiceProbability1; % If the randomnumber is below or equal the A Choice probability then A was choosen.
% If A isn't chosen, as it is a binary forced choice experiment, B must have been.
ChosenB=~Chosen1;  % The ~ sign means false, i.e. it gives a true/1 if Chosen A is false/0
SimulatedChoices1(:,isub)=Chosen1; % Now we just have to save the choices in the column of the agent in the loop
end


% As we don't actually know the exact subjective utility of our subjects,
% at least not immediately before running any models, we can also plot the 
% Softmax against objective utiltiy using our knowledge of the true
% underlying probabilities.

ActualUtilityA=trueProbabilityOpt1.*magOpt1;
ActualUtilityB=(1-trueProbabilityOpt1).*magOpt2;
DecisionVariableObjective=ActualUtilityA-ActualUtilityB;
 

% Now we can plot whether the choices we generated roughly follow the
% objective utility by binning A choices by the the objective Decision
% Variable. [Reminder: A perfect agent should have a step function at 0]
BinBorders=-80:20:80; % Lets go with -80 to 80 in steps of 20.
BinCentre=mean([BinBorders(1:end-1)' BinBorders(2:end)'],2); % Lets compute the centre of the bin so that we know where to plot.
for B=1:(length(BinBorders)-1)  % Loop over all Bins
    % Find what samples are included in the current bin using larger and
    % smaller as 
    CurrentIndex=repmat((DecisionVariableObjective>=BinBorders(B)) & (DecisionVariableObjective<BinBorders(B+1)),1,size(SimulatedChoices1,2)); 
    for a=1:size(SimulatedChoices1,2)
        BinValue(B,a)=mean(SimulatedChoices1(CurrentIndex(:,a),a)); % Compute the average choices for the current bin
    end
end
% Now plot the binned values at the currect locations in the middle of the bin boarders
figure('color',[1 1 1],'name','Actual choices homogenous simulated agent, Binned');
plot(repmat(BinCentre,1,size(BinValue,2)),BinValue,'.-');hold on;
errorbar(BinCentre,mean(BinValue,2),std(BinValue,0,2)./(size(BinValue,2).*.5),'k','Linewidth',1);
xlabel('objective Utility Difference (A-B)');ylabel('Probability of Choosing Option A');
set(gca,'Fontsize',16);




% Now that we have simulated many agents with exactly the same combination
% of Learning rate and temperature, we want to make sure that our analyses
% later are also sensitive when there is a distribution of both. In other
% words we are interested in later seeing the fitting recover both the
% average learning rates and temperatures for the agents as well as their
% variability around the group mean. This will ensure that our estimate of
% average model parameter is valid as well as any individual difference
% analysis we might want to do. To show you later how it looks when there
% are insufficient samples for recovering the true parameters from a
% simulation, we will also simulate a situation with far too few (or uninformative) trials.

Stable1stAversive  =load('stable_first_aversive_schedule.mat','trialVariables');
Stable1stNeutral   =load('stable_first_neutral_schedule.mat','trialVariables');
Volatile1stAversive=load('volatile_first_aversive_schedule.mat','trialVariables');
Volatile1stNeutral =load('volatile_first_neutral_schedule.mat','trialVariables');


% Of the 80 Samples we have 20 in each of the for schedule categories. For
% simplicity of structure, lets say the first 40 are stable first and the
% first 20 of stable and volatile first are in the aversive condition.
Stable1st        =[true(40,1);false(40,1)];
AversiveCondition=repmat([true(20,1);false(20,1)],2,1); % This just makes a string of 20 true, 20 false for Aversive condition and repeats it 
% (command: repmat(matrix,Rows,Columns))

% Now we need to draw learning rates and temperatures in order to test
% whether those values can be recovered from simulated agents with our
% schedule. To test whether estimates can recover the true underlying
% parameter values we might want to go with a sensible set of both
% temparatures and learning rates. In This examples lets take learning
% rates in between 0.001 and 1. Inverse Temperature will vary between 0.001
% and 1 for now
minPossibleLR=0.001;   %Define lowest possible Learning rate
maxPossibleLR=1;       %Define largest possible learning rate

sampledinverseTemps=log(1+abs(normrnd(0,.7,length(Stable1st),1)));

sampledLearningrates=normrnd(0.3,.2,length(Stable1st),1);   % Sample learning rates as a normal distribution with mean 0.3 and std of 0.7. 
 %Note, we don't know whether true learning rates will be normally distributed, but in the simulation we want to know whether we could recover
 %individual subject variance and for this normal distribution around mean
 %is a good working assumption

% Make sure Learning rates are bound between 0.001 and 1
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


% (Hopefully) good simulation
NrOfAgents=length(sampledinverseTemps) ;

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
    
    % Learn from outcomes with specified learning rate
    probOpt1(1,isub) = 0.5;
    for t = 1:numtrials-1
        delta(t,isub) = opt1Rewarded(t,isub)-probOpt1(t,isub);
        probOpt1(t+1,isub) = probOpt1(t,isub) + sampledLearningrates(isub)*delta(t,isub);
    end
    probOpt2(:,isub)=1-probOpt1(:,isub);       
    % Combine Probability and Magnitudes
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

 ActualUtilityA=trueProbabilityOpt1.*magOpt1;
ActualUtilityB=(1-trueProbabilityOpt1).*magOpt2;
DecisionVariableObjective=ActualUtilityA-ActualUtilityB;
       

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
ChoiceRewarded=opt1Rewarded==SimulatedChoices1;
subplot(2,1,2);
Stay=[nan(1,size(SimulatedChoices1,2));SimulatedChoices1(1:end-1,:)==SimulatedChoices1(2:end,:)];
LastTrialReward=[nan(1,size(ChoiceRewarded,2));ChoiceRewarded(1:end-1,:)];
for a=1:size(ChoiceRewarded,2)
    WinStay(a) =mean(Stay(LastTrialReward(:,a)==1,a));
    LoseStay(a)=mean(Stay(LastTrialReward(:,a)==0,a));
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
for isub=1:size(opt1Rewarded,2)
    for b=1:NrBack
        Pastopt1Rewarded(:,b,isub)=[nan(b,1);opt1Rewarded(1:end-b,isub)];
    end
end
DesignMatrixLabels={'t-1';'t-2';'t-3';'t-4';'t-5';'t-6';'t-7';'t-8';'t-9';'t-10'};
for isub=1:size(opt1Rewarded,2)
    % Take out?
    y=SimulatedChoices1(:,isub);
    DesignMatrix=Pastopt1Rewarded(:,:,isub);
    %
[betaweights(:,isub),devf,stats{isub}]=glmfit(DesignMatrix,y,'binomial','link','logit');
end
figure('color',[1 1 1],'name','Results of Multiple regression');
hold on;
plotData=betaweights(2:end,:);
bar(mean(plotData,2));hold on;%plot(plotData,'.','markersize',8);
errorbar(mean(plotData,2),std(plotData,0,2)./size(plotData,2).^.5,'.k');
%plot(plotData,'.','markersize',20)
set(gca,'XTick',1:length(DesignMatrixLabels),'XTickLabel',DesignMatrixLabels)
ylabel('effect size on choices (betaweights, arbitrary units)')
set(gca,'Fontsize',16);

