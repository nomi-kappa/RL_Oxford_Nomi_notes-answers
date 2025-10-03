function make_main_schedule(doSave)

%% HARD-CODED VARIABLES - determine the structure of the entire experiment

soundProbability = 0.4;            %p(sound outcome) being played on a given trial
latentStableProb = 0.75;            %p(reward) on high-probability option during Stable block
latentVolatileProb = 0.8;           %p(reward) on high-probability option during Volatile block
if nargin<1
    doSave = 0;
end


%% non-aversive, stable first

trialVariables = [];
trialVariables.isStableBlock(1:100,1) = 1;
trialVariables.isStableBlock(101:200,1) = 0;
trialVariables.magOpt1 = ceil(rand(200,1)*100);             %magnitude of reward on option 1
trialVariables.magOpt2 = ceil(rand(200,1)*100);             %magnitude of reward on option 2
trialVariables.trueProbability(1:100,1) = latentStableProb;
trialVariables.trueProbability(101:125,1) = 1-latentVolatileProb;      %latent probability of option 1 being rewarded
trialVariables.trueProbability(126:150,1) = latentVolatileProb;      %latent probability of option 1 being rewarded
trialVariables.trueProbability(151:175,1) = 1-latentVolatileProb;      %latent probability of option 1 being rewarded
trialVariables.trueProbability(175:200,1) = latentVolatileProb;      %latent probability of option 1 being rewarded
for i = 1:200
    trialVariables.opt1Rewarded(i,1) = rand<trialVariables.trueProbability(i);             %whether option 1 is rewarded on this trial
end

trialVariables.opt1Left = rand(200,1)>0.5;            %whether option 1 is presented on left or not
trialVariables.playsound = rand(200,1)<soundProbability * 1; %0 if no sound at outcome, 1=non-aversive, 2=aversive

trialVariables = reformat_trial_variables(trialVariables);
    
if doSave
    save('stable_first_neutral_schedule.mat');
end

%% aversive, stable first

for i = 1:200
    trialVariables(i).playsound = trialVariables(i).playsound * 2;
end
if doSave
    save('stable_first_aversive_schedule.mat');
end

%% non-aversive, volatile first

trialVariables = [];
trialVariables.isStableBlock(1:100,1) = 0;
trialVariables.isStableBlock(101:200,1) = 1;
trialVariables.magOpt1 = ceil(rand(200,1)*100);             %magnitude of reward on option 1
trialVariables.magOpt2 = ceil(rand(200,1)*100);              %magnitude of reward on option 2
trialVariables.trueProbability(1:25,1) = 1-latentVolatileProb;      %latent probability of option 1 being rewarded
trialVariables.trueProbability(26:50,1) = latentVolatileProb;      %latent probability of option 1 being rewarded
trialVariables.trueProbability(51:75,1) = 1-latentVolatileProb;      %latent probability of option 1 being rewarded
trialVariables.trueProbability(75:100,1) = latentVolatileProb;      %latent probability of option 1 being rewarded
trialVariables.trueProbability(101:200,1) = 1-latentStableProb;
for i = 1:200
    trialVariables.opt1Rewarded(i,1) = rand<trialVariables.trueProbability(i);             %whether option 1 is rewarded on this trial
end

trialVariables.opt1Left = rand(200,1)>0.5;            %whether option 1 is presented on left or not
trialVariables.playsound = rand(200,1)<soundProbability * 1; %0 if no sound at outcome, 1=non-aversive, 2=aversive

trialVariables = reformat_trial_variables(trialVariables);

if doSave
    save('volatile_first_neutral_schedule.mat');
end

%% aversive, volatile first

for i = 1:200
    trialVariables(i).playsound = trialVariables(i).playsound * 2;
end

if doSave
    save('volatile_first_aversive_schedule.mat');
end

end

function tVnew = reformat_trial_variables(trialVariables)

for i = 1:200
    tVnew(i).isStableBlock = trialVariables.isStableBlock(i);
    tVnew(i).magOpt1 = trialVariables.magOpt1(i);
    tVnew(i).magOpt2 = trialVariables.magOpt2(i);
    tVnew(i).trueProbability = trialVariables.trueProbability(i);
    tVnew(i).opt1Rewarded = trialVariables.opt1Rewarded(i);
    tVnew(i).opt1Left = trialVariables.opt1Left(i);
    tVnew(i).playsound = trialVariables.playsound(i);
end
end