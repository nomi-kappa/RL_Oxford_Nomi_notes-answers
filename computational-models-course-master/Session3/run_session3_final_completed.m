%% =================== Session 3 Description ======================== %
clear all;close all; % close any open figures and delete variables

% This is the main script for the third practical class.
% Here we are going to learn how to read in and do some basic plotting
% of the data you collected, how to fit data for each participant and do a
% model comparison at the end.
%
% Please refer to your workbook throughout


%% 3.0.1 Loading the data 

% All the data you collected was assembled into one matrix, so we can
% load it in the same way as we did for the simulated data in week 2.

% Adjust the basepath to your directory
basepath='C:/Users/mkleinflugge/Dropbox/Documents/Work/Documents/teaching/BlockPractical/Sess3_Miriam/';
load(fullfile(basepath,'data.mat'));
warning off;

%% 3.0.2 Inspecting individual participant's choices 
% Now you can plot some results from the data you collected.
% Let's plot an individual's data first, you can choose which one;
% try a few different participants and see if you can find some that tracked the
% probabilities really well and others that did not
plotSubj = 12;

% Go through this cell line by line and remind yourself what the
% different commands mean.
figure('color','w','name',['Participant',num2str(plotSubj)]);
plot(data.trueProbability(:,plotSubj),'k'); hold on;
plot(data.chosenOptionRewarded(:,plotSubj),'k*');
plot((data.opt1Chosen(:,plotSubj)+0.05).*0.9,'r.');
xlabel('Trials');
ylabel('Reward probability');
ylim([-0.1 1.1]);
legend('True Probability','Outcomes received','Choice');


%% 3.1.1 Calculating accuracy

% There is not just one way to do this but a good starting point is to look
% at some basic metrics such as the overall accuracy
% and see if any participants fall out of 3SD's of the mean

% Accuracy
% We will use a simple definition here which is whether the participant
% chose the option with the higher true underlying utility
% What would be a better way to do this?
% What does this logical statement do?
% Write down in your workbook in which cases it gives a '1' or a '0'
data.accuracy = (data.trueProbability.*data.magOpt1 > (1-data.trueProbability).*data.magOpt2) == data.opt1Chosen;
data.meanAccuracy = nanmean(data.accuracy,1);

%% 3.1.2 Plotting accuracy for all/stable/volatile trials
% Next we can look at accuracy for just the stable or volatile trials
% For that we calculate the mean again, but this time restricted to the
% stable and volatile trials as indicated in data.isStableBlock
data.accuracyStable = nan(size(data.accuracy));
data.accuracyVol  = nan(size(data.accuracy));
data.accuracyStable(data.isStableBlock==1) = data.accuracy(data.isStableBlock==1);
data.accuracyVol(data.isStableBlock~=1)  = data.accuracy(data.isStableBlock~=1);
data.meanAccuracyStable = nanmean(data.accuracyStable,1);
data.meanAccuracyVol  = nanmean(data.accuracyVol,1);

% A histogram ('hist') is a useful way to eyeball data because it shows how
% the data from all participants is distributed. We will now plot a
% histogram of the accuracy for stable, volatile and all trials
%
% Eyeball the data to see if anyone stands out from the crowd
% You can adjust the bins to make the plot more or less
% fine-grained and the xlims to see a smaller or larger range on the x axis
figure;
bins = [0.2:0.025:1]; xlims=[0,1.1];
subplot(3,1,1);hist(data.meanAccuracyStable,bins); xlim(xlims);title('Accuracy stable'); 
subplot(3,1,2);hist(data.meanAccuracyVol,bins); xlim(xlims);title('Accuracy volatile'); 
subplot(3,1,3);hist(data.meanAccuracy,bins); xlim(xlims);title('Mean accuracy');
xlabel('Accuracy');ylabel('nSubj');

% This final plot just puts the three accuracy measures into one matrix but with
% zscored values to see if any participant 'jumps out' - if someone has a
% very small or large accuracy compared to everyone else, this person would
% appear bright yellow or dark blue
AccZscored=[zscore(data.meanAccuracyStable);zscore(data.meanAccuracyVol);zscore(data.meanAccuracy)]';
figure;imagesc(AccZscored);colorbar;caxis([-3,3]);
title('Colormap for z-scored accuracy'); %TO BE ADDED BY STUDENT
xlabel('Block Type');ylabel('Subjects'); %TO BE ADDED BY STUDENT
set(gca,'XTick',[1 2 3],'XTickLabel',{'Stable';'Volatile';'Both'},'Fontsize',16); %TO BE ADDED BY STUDENT


%% 3.1.3 Plotting reaction times, mean and variance
% This cell plots the mean and std of the reaction time across trials for
% each participant, again this is mostly to eyeball the data and get a feel
% for everyone's performance

figure;
subplot(2,3,1:2);plot(mean(data.reactionTime));title('mean RT');ylabel('mean RT');ylim([0.5,2]);
subplot(2,3,4:5);plot(std(data.reactionTime));title('std RT');ylabel('std RT');ylim([0,8]);
xlabel('Subjects');

subplot(2,3,3);histogram(mean(data.reactionTime),[0.5:0.1:2],'Orientation','horizontal');ylim([0.5,2]);title('Histogram');
subplot(2,3,6);histogram(std(data.reactionTime),[0:0.5:8],'Orientation','horizontal');ylim([0,8]);xlabel('nSubj');

meanRTZscored = zscore(mean(data.reactionTime));
stdRTZscored = zscore(std(data.reactionTime));

%% 3.1.4 Identify outliers using the standard deviation

% Researchers can have reasons to be more or less strict in terms of which
% participants to exclude. One conservative threshold is to use three times
% the standard deviation (SD). 
discardThresh = 2.5;

% Define a participant as outlier if any of the above three criteria exceed 3*SD
% This line checks if the absolute zscore value is
% larger than our threshold above; it sums across criteria and then keeps
% all the participants that have a '1' in any of the criteria, i.e. where
% the sum is larger than 0
data.discardSubj = sum([abs(AccZscored) abs(meanRTZscored)' abs(stdRTZscored)']>discardThresh,2)>0;

% Add one line of code here to check how many subjects were excluded with the threshold we used
% the command 'find' might be useful which finds any element of a vector that
% is not zero
length(find(data.discardSubj)) %TO BE ADDED BY STUDENT

% This plots a red circle on top of the mean accuracy IF there is a
% subject that is an outlier
figure;plot(AccZscored);hold on;plot(meanRTZscored);plot(stdRTZscored);
plot(find(data.discardSubj),3*ones(length(find(data.discardSubj))),'ro');
title('All criteria for each subject (dots mark outliers)');
xlabel('Subjects');ylabel('zscore');
legend('Accuracy stable','Accuracy vol','Accuracy all','mean RT','std RT','outliers');


%% 3.1.5 Win-stay/Lose-shift
% This is code copied from session 2 and you do not need to read through it
% line by line. 

% Here it first calculates whether subjects stayed after a win or loss
Stay=[nan(1,size(data.opt1Chosen,2));data.opt1Chosen(1:end-1,:)==data.opt1Chosen(2:end,:)];
LastTrialReward=[nan(1,size(data.chosenOptionRewarded,2));data.chosenOptionRewarded(1:end-1,:)];
for a=1:size(LastTrialReward,2)
    WinStay(a) = mean(Stay(LastTrialReward(:,a)==1,a));
    LoseStay(a)= mean(Stay(LastTrialReward(:,a)==0,a));
    WinStayVol(a) = mean(Stay(LastTrialReward(:,a)==1 & data.isStableBlock(:,a)==0,a));
    LoseStayVol(a)= mean(Stay(LastTrialReward(:,a)==0 & data.isStableBlock(:,a)==0,a));
    WinStayStb(a) = mean(Stay(LastTrialReward(:,a)==1 & data.isStableBlock(:,a)==1,a));
    LoseStayStb(a)= mean(Stay(LastTrialReward(:,a)==0 & data.isStableBlock(:,a)==1,a));
end

%  Plot win/stay on average across stable and volatile
figure;hold on;
subplot(1,3,1);hold on;bar([mean(WinStay ) mean(LoseStay )],'FaceColor',[0.5 0.5 0.5]);
errorbar([mean(WinStay ) mean(LoseStay )],([std(WinStay ) std(LoseStay )]./(length(WinStay).^.5)),'.k','Linewidth',4);
plot([WinStay' LoseStay']','k-');xlim([0.5 2.5]);ylim([0.3,1]);
set(gca,'XTick',[1 2],'XTickLabel',{'Win Stay';'Lose Stay'},'Fontsize',16)
ylabel('Probability of Staying');title('All');

% Plot win/stay separately for stable and volatile
subplot(1,3,2);hold on;bar([mean(WinStayVol) mean(LoseStayVol)],'FaceColor',[0.5 0.5 0.5]);
errorbar([mean(WinStayVol) mean(LoseStayVol)],([std(WinStayVol ) std(LoseStayVol)]./(length(WinStayVol).^.5)),'.k','Linewidth',4);
plot([WinStayVol' LoseStayVol']','k-');xlim([0.5 2.5]);ylim([0.3,1]);
set(gca,'XTick',[1 2],'XTickLabel',{'Win Stay';'Lose Stay'},'Fontsize',16)
ylabel('Probability of Staying');title('Volatile');

subplot(1,3,3);hold on;bar([mean(WinStayStb) mean(LoseStayStb)],'FaceColor',[0.5 0.5 0.5]);
errorbar([mean(WinStayStb) mean(LoseStayStb)],([std(WinStayStb) std(LoseStayStb)]./(length(WinStayStb).^.5)),'.k','Linewidth',4);
plot([WinStayStb' LoseStayStb']','k-');xlim([0.5 2.5]);ylim([0.3,1]);
set(gca,'XTick',[1 2],'XTickLabel',{'Win Stay';'Lose Stay'},'Fontsize',16)
ylabel('Probability of Staying');title('Stable');

% If you are curious, you can do some very basic t-test here...
        %Since both vectors have one value per participant, MATLAB compares them within subjects
[p,h,stats]=ttest(WinStayVol,WinStayStb)

% mean(WinStayVol - WinStayStb) = -0.0685 % participants stayed slightly less in volatile blocks



%% 3.2 MODEL FITTING USING GRID SEARCH
%% 3.2.1 Developing an intuition for the goodness of a fit
% We will try and first get an intuition of what a good fit and a bad fit
% look like. For that, let's start by choosing any participant
subjN = 12;

% For this cell to work you need to be in the folder that contains
% RLModel.m (check), or do addpath(FOLDER) where the FOLDER is the one that
% contains the RLModel.m file

% We are going to use the function RLModel which needs as inputs 
% the learning rate alpha and the inverse temperature parameter beta, 
% the subject you are fitting, the data, and whether you want to plot 
% the predicted choice probability (1=yes); it gives back an error term,
% the smaller this error term, the better the fit.
%
% Let's run it twice with different alpha & beta values just to visualize 
% the results of the RLmodel. Note down which sets of parameters you think 
% fits this participant better -- see your notebook.
error = RLModel([0.15,0.02],subjN,data,1)
error = RLModel([0.6,0.08],subjN,data,1)

% If you want to try using two different learning rates for the stable 
% and volatile phase, you can do that by giving RLmodel three parameter 
% values (the first two are alpha stable/alpha volatile, third one is beta)
% Just uncomment the line below...
% error = RLModel([0.1,0.3,0.1],subjN,data,1)


%% 3.2.2 Grid search 
% Now, let us try and determine the best-fitting learning rate and 
% inverse temperature parameter using a grid search. In other words, 
% we define a grid of values for LR and invT and see which 'error' value 
% we get back from the RL-learner. Our aim is to find the combination of 
% parameters that gives the smallest error, or in other words, best predicts 
% the choices.

% You can choose a subject here
subjN = 1;

% These two lines of code determine which values form part of the grid
% search. You can make the grid finer by making the step sizes smaller or
% coarser by making them larger (and thus including less values).
gridLearningRate = [0.1:0.1:1];%[0.05:0.05:1];
gridInverseT = [0.01:0.03:0.3];

% Now we run two for-loops: one over all alpha and one over all beta
% starting values, and we save the logLL (error term) for each.
logLLGrid = zeros(length(gridLearningRate),length(gridInverseT));
for a=1:length(gridLearningRate)
    LR=gridLearningRate(a);
    for b=1:length(gridInverseT)
        invT=gridInverseT(b);
        logLLGrid(a,b)= RLModel([LR,invT],subjN,data,0);
    end
end

% This figure just plots the error values we got for all combinations of
% alpha & beta in thw two for loops above as a colour scale
% imagesc = Scale data and display as image
figure;imagesc(logLLGrid);
ylabel('Learning rate (alpha)');xlabel('Inverse softmax temperature (beta)');
title(['Subject ',num2str(subjN)]);colorbar;
set(gca,'XTick',[1:length(gridInverseT)],'XTickLabel',gridInverseT)
set(gca,'YTick',[1:length(gridLearningRate)],'YTickLabel',gridLearningRate)



%% 3.2.3 Grid search: separately for volatile & stable
% Here we do a gridsearch again but separately on stable & volatile blocks
% This is automatically taken care of internally by RLModel.m when more 
% than two parameters are given to the function.
gridLearningRate = [0.1:0.1:1];
gridInverseT = [0.01:0.03:0.3];

subjN=12;

% So now we need two learning rates and two betas. Here we define all
% possible combinations for our grid search, this is a bit easier than
% stacking three or four for-loops into each other
parast = [repmat({gridLearningRate},1,2) repmat({gridInverseT},1,2)];
params = combvec(parast{1},parast{2},parast{3},parast{4})';

% Then we run and plot the model with all combinations of parameters
logLLGrid = zeros(size(params,1),1);
for pCombi = 1:size(params,1)
    logLLGrid(pCombi)=RLModel(params(pCombi,:),subjN,data,0);
end

% logLLGrid now has as many 'error values' as there are parameter
% combinations in params.
% Insert code here to find out the parameter values that correspond to the 
% minimum logLL
minVal = find(logLLGrid==min(logLLGrid)); %TO BE ADDED BY STUDENT
params(minVal,:) %TO BE ADDED BY STUDENT

% This visualizes the grid again for stable and volatile
% Don't worry too much about understanding every single line of the code at
% this point
logLLGrid = reshape(logLLGrid,length(gridInverseT),length(gridInverseT),length(gridLearningRate),length(gridLearningRate));
params = reshape(params,length(gridLearningRate),length(gridLearningRate),length(gridInverseT),length(gridInverseT),size(params,2));
figure;
subplot(1,2,1);
imagesc(squeeze(min(min(logLLGrid,[],3),[],1))); title('Stable');
set(gca,'XTick',[1:length(gridInverseT)],'XTickLabel',gridInverseT)
set(gca,'YTick',[1:length(gridLearningRate)],'YTickLabel',gridLearningRate)

subplot(1,2,2);
imagesc(squeeze(min(min(logLLGrid,[],4),[],2))); title('Volatile');
xlabel('Alpha value');ylabel('Beta value');
set(gca,'XTick',[1:length(gridInverseT)],'XTickLabel',gridInverseT)
set(gca,'YTick',[1:length(gridLearningRate)],'YTickLabel',gridLearningRate)


%% 3.2.4 Grid search for all subjects, separately for volatile & stable
% This is the same code as in 3.2.3 except that we do the grid search
% across all subjects now
% This will take a moment to run; the grid is coarser to help with speed
gridLearningRate = [0.1:0.2:0.9];
gridInverseT = [0.01:0.06:0.3];
parast = [repmat({gridLearningRate},1,2) repmat({gridInverseT},1,2)];
params = combvec(parast{1},parast{2},parast{3},parast{4})';
nSub = size(data.discardSubj,1);
logLLGrid = zeros(nSub,size(params,1),1);
for s=1:nSub
    for pCombi = 1:size(params,1)
        logLLGrid(s,pCombi)=RLModel(params(pCombi,:),s,data,0);
    end     
    % find best fit
    minVal(s) = find(logLLGrid(s,:)==min(logLLGrid(s,:))); 
    bestParams(s,:) = params(minVal(s),:);
end

% This makes a figure with four subplots that shows a histogram for each of the
% parameter estimates:
titles = {'LR stable','LR volatile','beta stable','beta volatile'};
figure;
for i=1:size(bestParams,2)
    subplot(2,2,i);hist(bestParams(:,i),[0:0.2:1]);xlim([0,1]);title(titles{i});
end

% Please insert here one line that saves the mean of the best parameter
% estimates across subjects; we can then use that mean to initialize 
% fminsearch in the next cell
meanBestParam = mean(bestParams,1); %TO BE ADDED BY STUDENT

%% 3.3 MODEL FITTING USING FMINSEARCH
%% 3.3.1 Fitting two learning rates and one beta
% A grid search is difficult to do with too many parameters and can be 
% quite inefficient; Matlab has a function called fminsearch that
% can look for minima of any function without having to measure every 
% single point

% First we need an initialization for our parameters - at the moment this
% uses one initialization from the average of the grid search, and one at
% the lower and higher end of values (i.e. three values per parameter)
alphaInit  = [0.1,mean(meanBestParam(1:2)),0.6];%[0.1 0.2]; % learning rates
betaInit   = [0.01,mean(meanBestParam(3:4)),0.1];%[0.5 3];   % inverse temperature
parast = [repmat({alphaInit},1,2) repmat({betaInit},1,1)];
params = combvec(parast{1},parast{2},parast{3})';
nSub = size(data.discardSubj,1);

% Then we specify how many times the function should be maximally evaluated
% and how many iterations to run maximally
opts.MaxFunEvals=10000;
opts.MaxIter    =10000;

% Run the actual model fits
for s=1:nSub
    % run the minimization
    for ip = 1:size(params,1)
        [parFitAll(s,ip,:) negLogLL(s,ip)] = fminsearch(@RLModel,params(ip,:),opts,s,data,0);  % store these values to later select the best fitting parameters
    end
    
    % Find the parameter initialization that minimizes the errorterm
    minErr = find(negLogLL(s,:)==min(negLogLL(s,:)));
   
    % Save the paramters and error for that best combination of starting
    % values
    parFit3(s,:)= squeeze(parFitAll(s,minErr,:));
    negLogLL3(s,:)  = negLogLL(s,minErr);
end

% Re-save data and include new fitting information in it
% This will be used in block practical four for all statistical tests
data.LearningRateStable = parFit3(:,1);
data.LearningRateVolatile = parFit3(:,2);
data.InverseTemperature3 = parFit3(:,3);
save('dataFitted','data');

% Plot a simple bargraph of the learning rates in stable and volatile
figure;hold on;
bar([mean(data.LearningRateStable ) mean(data.LearningRateVolatile)],'FaceColor',[0.5 0.5 0.5],'BarWidth',0.5);
errorbar([mean(data.LearningRateStable ) mean(data.LearningRateVolatile)],([std(data.LearningRateStable ) std(data.LearningRateVolatile)]./sqrt(size(data.LearningRateStable,1))),'.k','Linewidth',4);
plot([data.LearningRateStable data.LearningRateVolatile]','k-');ylim([0,1]);
title('Learning rate');
set(gca,'XTick',[1 2],'XTickLabel',{'Stable';'Volatile'},'Fontsize',14)

[p,h,stats]=ttest(data.LearningRateStable,data.LearningRateVolatile)


%% 3.4. MODEL COMPARISON
%% 3.4.1 Fitting one learning rate and one beta
% This code is identical to the previous cell but instead of calling
% RLmodel with two alphas and one beta, we now just give it one alpha and
% one beta. Otherwise all is the same...

clear parFitAll negLogLL;
alphaInit  = [0.1,mean(meanBestParam(1:2)),0.6];%[0.1 0.2]; % learning rates
betaInit   = [0.01,mean(meanBestParam(3:4)),0.1];%[0.5 3];   % inverse temperature
parast = [repmat({alphaInit},1,1) repmat({betaInit},1,1)]; % THE ONLY CHANGE IS IN THIS LINE (with respect to the previous cell)
params = combvec(parast{1},parast{2})';
nSub = size(data.discardSubj,1);

opts.MaxFunEvals=10000;
opts.MaxIter    =10000;

% Run the actual model fits
inclSubj = find(~data.discardSubj);
for s=1:nSub
    % run the minimization
    for ip = 1:size(params,1)
        [parFitAll(s,ip,:) negLogLL(s,ip)] = fminsearch(@RLModel,params(ip,:),opts,s,data,0);  % store these values to later select the best fitting parameters
    end
    
    % Find the parameter initialization that minimizes the errorterm
    minErr = find(negLogLL(s,:,:)==min(negLogLL(s,:)));
   
    % Save the paramters and error for that best combination of starting
    % values
    parFit2(s,:)= squeeze(parFitAll(s,minErr,:));
    negLogLL2(s,:)= negLogLL(s,minErr);
end

% Again save these parameters for later use
data.LearningRateBoth = parFit2(:,1);
data.InverseTemperature2 = parFit2(:,2);
save('dataFitted','data');


%% 3.4.2 Computing the AIC for model comparison
% Here we will compute the Akaike Information Criterion (AIC). Computing
% the AIC values is really simple:
% We need to define the log likelihood which is the negative of the
% negative loglikelihood we got back from fminsearch and we need to know
% the number of parameters in our model, which was 2 and 3, respectively.
logLL2 = -negLogLL2; %log-likelihood of 2-param model
logLL3 = -negLogLL3; %log-likelihood of 3-param model

% AIC for the 2-parameter model
numParam=2;
AIC2 = -2*sum(logLL2) + 2*numParam;

% AIC for the 3-parameter model
numParam=3;
AIC3 = -2*sum(logLL3) + 2*numParam;

% Smaller is better: have a look at the AIC value we get on average
fprintf(['AIC values: ',num2str([AIC2 AIC3]),'\n']);

% Let's plot the AIC difference for all participants
figure;bar([-2*logLL3 + 2*3]-[-2*logLL2 + 2*2]);
title('AIC for 3param-2param (neg: 3param is better; pos: 2param better)');

% How much better is the 3-param compared to the 2-param model?
% We can compute its relative likelihood 
% This formula desribes how much more likely the 2-param model is (compared
% to the 3-param model) to minimize information loss/be the better model
% If this number if small, three parameters are definitely better
% If this number is bigger, both models are worth considering/we cannot
% decide between them...
relLL3=exp((AIC3-AIC2)/2);
fprintf(['Relative likelihood: ',num2str(relLL3),'\n']);

% If you have time left, you could write code here that does the AIC
% comparison for the people who encountered stable first or the people who
% encountered volatile first
