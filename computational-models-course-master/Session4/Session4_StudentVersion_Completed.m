%% Read data
clear
basepath = '/Users/JScholl/Documents/Teaching/CompBlockPractical/MatlabCode/Session4';
addpath(genpath(basepath)); % This command allows Matlab to find and access all data files and scripts in the folder

% in this matrix, write down which participants you want to exclude
%excludeParticipants = [16 26 33 56]; % format: [1, 10, 20] etc. 
%excludeParticipants = [1 3 5 7 8 13 21 27 30 34 38 44 48 58]; % reg2
excludeParticipants = [1 8 34 35 38 61]; % reg3
% Load the real data
load(fullfile(basepath,'RealDataFitted'))
RealData = StoreData;  
load(fullfile(basepath,'RealData'))
RealData.OriginalData = data;
% Load the simulated data
load(fullfile(basepath,'SimulationOneLearningRateFitted'))
SimulData.OneLearningRate = StoreData;
load(fullfile(basepath,'SimulationTwoLearningRatesFitted'))
SimulData.TwoLearningRates = StoreData;

clear StoreData % let's delete this so we don't accidentally use it later

% To simplify the code later (because e.g.
% RealData.TwoLearningRatesModel.Parameters.learningRateStable is very
% long), we will load the data here and store it in variables with simpler
% names 

% Find the participants we want to include
InclSubs = ones(length(RealData.GroupIndex.value),1);
InclSubs(excludeParticipants) = 0; % remove the excluded participants
InclSubs= logical(InclSubs);

% Find the participants in the control or the stress group (without the
% ones that we want to exclude)
Controls = RealData.GroupIndex.value==1 & InclSubs==1;
Stress   = RealData.GroupIndex.value==2 & InclSubs==1;

% Extract the learning rates matrix: rows are the subjects and columns:
% learning rate in stable, then learning rate in volatile block
Alphas = [RealData.TwoLearningRatesModel.Parameters.learningRateStable RealData.TwoLearningRatesModel.Parameters.learningRateVolatile]; 
Alphas = (Alphas);  % modify this line if you want to transform your data

% Extract the model fits
AICs = [RealData.OneLearningRateModel.AIC RealData.TwoLearningRatesModel.AIC]; 

% Participants' ratings
RatingsHappy  = likertData.happy';
RatingsStress = likertData.stress';
% Decide here which rating you want to use for the later analyses - either
% the question about happiness or stress (or even a combination of the two)
% - if you change it to RatingsHappy, don't forget to later change the
% figure legends and labels and make a note of what exactly you are showing
% in your report
Ratings=RatingsStress;

% count the number of valid subjects in each group (we compute it here
% because we will need this number several times later, e.g. to compute the
% standard error
nsubContr = sum(Controls);
nsubStress= sum(Stress);


subJnumber = Controls+2*Stress; % to have indexing for JASP (see optional question)

%% A.0 Model comparisons (AIC)
% This should produce figures like in session 3, we're just repeating this
% here so that you have all figures in one place
figure('color','w')
bar(AICs(InclSubs,2)-AICs(InclSubs,1));
title('AIC for 2 learning rates model minus AIC for 1 learning rate model')
ylabel('AIC difference')
xlabel('Participant')

figure('color','w')
subplot(2,1,1)
bar(AICs(Controls,2)-AICs(Controls,1));
title('AIC difference for control participants')
ylabel('AIC difference')
xlabel('Participant')

subplot(2,1,2)
bar(AICs(Stress,2)-AICs(Stress,1));
title('AIC difference for participants in stress condition')
ylabel('AIC difference')
xlabel('Participant')
%% A.1 Bar plots stable vs. volatile
% Plot bar graphs separately for the two groups and separately for each
% condition
figure('color','w')
subplot(2,2,1)
% extract the learning rates from the two conditions and compute the means
% and standard errors
means = [mean(Alphas(Controls==1,:)); mean(Alphas(Stress==1,:))]; % mean values in each condition and in each group
stes  = [std(Alphas(Controls==1,:))./sqrt(nsubContr); std(Alphas(Stress==1,:))./sqrt(nsubStress)]; % compute the standard error for each group and condition, i.e. the standard deviation divided by the square root of the number of participants
% Plot the bars:
bar(means)
hold on
% Add the error bars:
barPositions = [0.85 1.15;1.85 2.15]; % where the different bars should be on the graph
errorbar(barPositions,means,stes,'k.')
% to see the single subjects we can add them on top
plot(([0.85 1.15].*ones(nsubContr,2))',Alphas(Controls==1,:)')
plot(([1.85 2.15].*ones(nsubStress,2))',Alphas(Stress==1,:)')

set(gca,'XTickLabel',{'Control','Stress'})
legend('Stable','Volatile')
ylabel('Learning Rate')
title('Bar graphs of learning rates')
set(gca,'FontSize',12)

% Plot bar graphs for the difference between the conditions
% let's first compute the difference volatile minus stable:
AlphaVolMinStable = Alphas(:,2)-Alphas(:,1);
means = [mean(AlphaVolMinStable(Controls==1)); mean(AlphaVolMinStable(Stress==1))]; % mean values in each condition and in each group
stes  = [std(AlphaVolMinStable(Controls==1))./sqrt(nsubContr); std(AlphaVolMinStable(Stress==1))./sqrt(nsubStress)]; % compute the standard error for each group and condition, i.e. the standard deviation divided by the square root of the number of participants
% Plot the bars:
subplot(2,2,3)
bar(means)
hold on
% Add the error bars:
%barPositions = [0.85 1.15;1.85 2.15]; % where the different bars should be on the graph
errorbar(means,stes,'k.')
set(gca,'XTickLabel',{'Control','Stress'})
%legend('Stable','Volatile')
ylabel('Learning Rate Volatile minus stable')
title('Learning rates volatile minus stable')
set(gca,'FontSize',12)

% We can look at the same data as box plots to check for outliers (box
% plots have a slightly funny way they take data, we have to make one long
% vector with all the data and then another vector of indices that says
% which type each entry is 
combDat = [Alphas(:,1);Alphas(:,2)];
indices = [1*Controls+2*Stress;3*Controls+4*Stress]; %1= controls and stable, 2=stress and stable, 3= controls and volatile, 4= stress and volatile 
subplot(2,2,2)
boxplot(combDat(indices~=0), indices(indices~=0),'Labels',{'Contr. Stable','Stress Stable','Contr. Vol.','Stress Vol.'})
title('Box plots of learning rates')
ylabel('Learning rate')
set(gca,'FontSize',12)

% Box plots for the differences
indices = 1*Controls+2*Stress;
subplot(2,2,4)
boxplot(AlphaVolMinStable(indices~=0), indices(indices~=0),'Labels',{'Controls','Stress'})
title('Box plots of learning rate difference')
ylabel('Learning rate difference')
set(gca,'FontSize',12)


%% A.2 Histograms
% Plot histograms: separately for each group, plot the learning rates, as
% well as the difference between the stable and the volatile block

nbins = 8; %number of bins for the histogram

% Histograms for each group, combined across both conditions
figure('color','w')
subplot(2,2,1)
histogram(Alphas(Controls,:),nbins) % have a look 'doc histogram' to get information about the histogram settings 
title('Learning rate control condition')
ylabel('Number of participants')
xlabel('Learning rate')

subplot(2,2,2)
histogram(Alphas(Stress,:),nbins)
title('Learning rate stress condition ')
ylabel('Number of participants')
xlabel('Learning rate')

% Histograms of the differences in learning rates between the stable and
% volatile blocks
subplot(2,2,3)
histogram(AlphaVolMinStable(Controls),nbins)
title('Learning rate difference controls')
ylabel('Number of participants')
xlabel('Learning rate diff.')

subplot(2,2,4)
histogram(AlphaVolMinStable(Stress),nbins)
title('Learning rate difference stress cond.')
ylabel('Number of participants')
xlabel('Learning rate diff.')


%% A.4 Paired T-tests
display('Within subject t-test comparing the learning rates in the stable vs. volatile block across the two groups');

% QA.4 - remove the code below
[h,p,ci,stats]=ttest(Alphas(InclSubs,1),Alphas(InclSubs,2))

% compute the effect size
% QA.4 - remove some of the code below
effectSizeStats = mes(Alphas(InclSubs,1),Alphas(InclSubs,2),'hedgesg','isDep',1,'nBoot',10000) % look at doc mes to see what the inputs mean

% DELTE HERE
[p,h] = signrank(Alphas(InclSubs,1),Alphas(InclSubs,2))
effS = mes(Alphas(InclSubs,1),Alphas(InclSubs,2),'U1','isDep',1,'nBoot',10000)

%% B.1 Checking the stress manipulation
% Let's visualize the data: Ratings over time
figure('color','w','name','stress ratings')
subplot(2,1,1)
% Plot the controls
h1=plot(mean(Ratings(Controls,:)),'b','LineWidth',2);%'b' tells Matlab to plot in blue
hold on
h2=plot(Ratings(Controls,:)','b--');  %Add the data from single subjects; 'b--' means 'plot in blue with dashed line'
% Plot the participants in the stress group
h3=plot(mean(Ratings(Stress,:)),'r','LineWidth',2);
h4=plot(Ratings(Stress,:)','r--');
xlabel('Rating time point')
ylabel('Rating')
title('Stress ratings over time')
legend([h1 h3],'Controls','Stress group')
set(gca,'FontSize',12)
xlim([0.5 8.5])
% Checking the stress manipulation by comparing the average stress ratings
% Compute the average ratings
avgRating = mean(Ratings,2);

% Check the distributions
subplot(2,1,2)
histogram(avgRating(Controls),8,'FaceColor','b')
hold on
histogram(avgRating(Stress),8,'FaceColor','r')
xlabel('Average Rating')
ylabel('Number of participants')
legend('Controls','Stress group')
title('Histogram of average ratings')
set(gca,'FontSize',12)

% Do an independent samples t-test to 
display('Between subject t-test comparing the average ratings between the two groups');
[h,p,ci,stats] = ttest2(avgRating(Controls),avgRating(Stress))
effectSizeStats = mes(avgRating(Controls),avgRating(Stress),'hedgesg','nBoot',10000)

[pRankSum, hRankStum, statsRankSum] = ranksum(avgRating(Controls),avgRating(Stress))
effectSizeStats = mes(avgRating(Controls),avgRating(Stress),'U1','nBoot',10000)


%% B.2 Does stress affect the ability to adapt learning rates to the environment?
% Write your own code here to do a t-test comparing the adaptation in
% learning rates (AlphaVolMinStable) between the two groups
% DELTE THIS CODE!!!
display('Between subject t-test comparing the average ratings between the two groups');
[h,p,ci,stats] = ttest2(AlphaVolMinStable(Controls),AlphaVolMinStable(Stress))
effectSizeStats = mes(AlphaVolMinStable(Controls),AlphaVolMinStable(Stress),'hedgesg')

%% C.Correlation between average stress and adaptation of learning rates
figure('color','w')
% Make scatter plots, showing the controls and the stress group data in
% diferent colors
subplot(2,1,1)
s1=scatter(AlphaVolMinStable(Controls),avgRating(Controls),'filled','MarkerFaceColor','b');
hold on
s2=scatter(AlphaVolMinStable(Stress), avgRating(Stress),'filled','MarkerFaceColor','r');



% identify the reationc and ratings outliers (of course only for
% participants that we are actually including in the analysis)
stdThresh=2.5;
outliersLRdiffs = abs(zscore(AlphaVolMinStable))> stdThresh & InclSubs==1;
outliersAvgRatings = abs(zscore(AlphaVolMinStable))>stdThresh & InclSubs==1;
% Highlight the outliers with a diamond shape
s3=scatter(AlphaVolMinStable(outliersLRdiffs),avgRating(outliersLRdiffs),100,'diamond','k','LineWidth',1);
legend([s1 s2 s3] ,'Controls','Stress group','Outliers')
ylabel('avg. Rating')
xlabel('Difference in learning rate')
set(gca,'FontSize',12)



% Plot least square lines with and without outliers
subplot(2,1,2)
s4=scatter(AlphaVolMinStable(InclSubs),avgRating(InclSubs),'filled','MarkerFaceColor','g');
hold on
nonOutliers = InclSubs==1 & outliersLRdiffs==0 & outliersAvgRatings==0;
s5=scatter(AlphaVolMinStable(nonOutliers),avgRating(nonOutliers),'filled','MarkerFaceColor','k');
lss=lsline
set(lss(2),'color','g')
set(lss(1),'color','k') 
legend([s1 s2],'Controls','Stress group')
ylabel('avg. Rating')
xlabel('Difference in learning rate')
set(gca,'FontSize',12)
legend([lss(1) lss(2)],'Without outliers','With outliers')

% Compute the correlation
display('Correlation between learning rate difference and average rating');
[rho, pval] = corr(AlphaVolMinStable(InclSubs),avgRating(InclSubs))


%% D.1 Extra plots to help you write your report - No need to look at it in session 4. 
%Plotting a figure to illustrate the schedule
% Plot the schedule for illustration - this could be a figure in your write
% up
figure('color','w')
% Plot the probabilities
subplot(2,1,1)
plot(RealData.Schedule.trueProbability(:,1),'k','LineWidth',2)
hold on
plot(RealData.Schedule.opt1Rewarded(:,1),'*k')
% Complete the following lines to label your graph (hint: look at what we
% are plotting with the 'plot' command)
legend('','')
ylim([-0.1 1.1])
xlabel('')
ylabel('')
title('')

% Plot the magnitudes
subplot(2,1,2)
plot([RealData.Schedule.magOpt1(:,1) RealData.Schedule.magOpt2(:,1)]);
% Complete the lines to label your graph
legend('','')
xlabel('')
ylabel('')
title('')
%% D.2 Extra plots to help you write your report - No need to look at it in session 4. 
% Plots to check the model simulations
% You could choose to include all or only some of the plots made in this
% cell in your report

% 1) Plot a scatter plot of the real and simulated data (when fitted with the
% model using two learning rates) - for learning rates and inverse
% temperatue

% For this we need to first put the data from the two simulations together
% Put all the simulated learning rates together
simulatedLearningRates = [SimulData.OneLearningRate.OriginalParameters.learningRate;SimulData.OneLearningRate.OriginalParameters.learningRate; ...
    SimulData.TwoLearningRates.OriginalParameters.learningRateStable;SimulData.TwoLearningRates.OriginalParameters.learningRateVolatile];
% this produces a long vector with all simulated learning rates
% (independent of whether they were simulated from a model using a single
% learning rate for two blocks or separate learning rates)

% Put all the learning rates fitted with the model assuming two learning
% rates together
fittedLearningRates = [SimulData.OneLearningRate.TwoLearningRatesModel.Parameters.learningRateStable; SimulData.OneLearningRate.TwoLearningRatesModel.Parameters.learningRateStable;...
    SimulData.TwoLearningRates.TwoLearningRatesModel.Parameters.learningRateStable;SimulData.TwoLearningRates.TwoLearningRatesModel.Parameters.learningRateVolatile];

% Calculate the difference in simulated learning rates between the two
% blocks (for the model with only one learning rate, this difference is
% zero) 
simulatedLRdifference = [zeros(size(SimulData.OneLearningRate.OriginalParameters.learningRate));...
    (SimulData.TwoLearningRates.OriginalParameters.learningRateVolatile-SimulData.TwoLearningRates.OriginalParameters.learningRateStable)];
% Calculate the difference in the fitted learning rates (volatile minus
% stable)
fittedLRdifference = [SimulData.OneLearningRate.TwoLearningRatesModel.Parameters.learningRateVolatile-SimulData.OneLearningRate.TwoLearningRatesModel.Parameters.learningRateStable; ...
    SimulData.TwoLearningRates.TwoLearningRatesModel.Parameters.learningRateVolatile-SimulData.TwoLearningRates.TwoLearningRatesModel.Parameters.learningRateStable];

% Put the simulated inverse temperatures together
simulatedInvTemp = [SimulData.OneLearningRate.OriginalParameters.inverseTemperature;SimulData.TwoLearningRates.OriginalParameters.inverseTemperature];
% Put all the learning rates fitted with the model assuming two learning
% rates together
fittedInvTemp = [SimulData.OneLearningRate.TwoLearningRatesModel.Parameters.inverseTemperature;SimulData.TwoLearningRates.TwoLearningRatesModel.Parameters.inverseTemperature];

% Let's make scatter plots of these quantities
figure('color','w','name','Check parameter recovery')
subplot(1,3,1)
scatter(simulatedLearningRates,fittedLearningRates)
xlabel('')
ylabel('')
subplot(1,3,2)
scatter(simulatedInvTemp,fittedInvTemp)
xlabel('')
ylabel('')
subplot(1,3,3)
scatter(simulatedLRdifference,fittedLRdifference)
xlabel('')
ylabel('')
% For you to add: compute statistics (r-value of the correlations), as
% described for linking learning rate differences to mood in the other
% parts of session 4

% Plot the model that was simulated as having one learning rate when
% analysed with the model that tries to fit two learning rates; and plot
% the model that was simulated as having two learning rates when analysed
% with the model that tries to fit two learning rates
figure('color','w')
subplot(2,1,1)
LRstable = SimulData.OneLearningRate.TwoLearningRatesModel.Parameters.learningRateStable;
LRvolatile= SimulData.OneLearningRate.TwoLearningRatesModel.Parameters.learningRateVolatile;
meanLRs = mean([LRstable LRvolatile]); % mean of the learning rates
steLRs  = std([LRstable LRvolatile])./sqrt(size(LRstable,1)); % standard error across simulated subjects
bar(meanLRs)
hold on
errorbar(meanLRs, steLRs,'.k')
ylabel('')
set(gca,'XTickLabel',{'',''})
title('')

subplot(2,1,2)
LRstable = SimulData.TwoLearningRates.TwoLearningRatesModel.Parameters.learningRateStable;
LRvolatile= SimulData.TwoLearningRates.TwoLearningRatesModel.Parameters.learningRateVolatile;
meanLRs = mean([LRstable LRvolatile]); % mean of the learning rates
steLRs  = std([LRstable LRvolatile])./sqrt(size(LRstable,1)); % standard error across simulated subjects
bar(meanLRs)
hold on
errorbar(meanLRs, steLRs,'.k')
ylabel('')
set(gca,'XTickLabel',{'',''})
title('')

% for you to do: compute statistics

%% D.3 Using regression analy to check whether participants took all features of the task into account
% We will run a logistic regression separately for each person to predict
% their choice on each trial as a function of the true reward probability
% and the reward magnitudes of the two options. 
% We will store the regression weights in a matrix that we first initialize
regWeights = nan(size(RealData.OriginalData.opt1Chosen,2),4);
for is = 1:size(RealData.OriginalData.opt1Chosen,2) % make a for loop that runs a regression for each subject
    % extract the relevant data (for that person) and give it shorter variable names
    trueProb = RealData.OriginalData.trueProbability(:,is);
    magOpt1  = RealData.OriginalData.magOpt1(:,is);
    magOpt2  = RealData.OriginalData.magOpt2(:,is);
    choices  = RealData.OriginalData.opt1Chosen(:,is);
    % put all the predictors together

    allPredictors = zscore([trueProb magOpt1 magOpt2 ]); 
    regWeights(is,:) = glmfit(allPredictors,choices,'binomial');
end

figure('color','w')
bar(mean(regWeights(:,2:end)))
hold on
plot(regWeights(:,2:end)','x')
ylabel('Regression weight')
set(gca,'XTickLabel',{'Prob','MagOpt1','MagOpt2'},'XTickLabelRotation',45)
