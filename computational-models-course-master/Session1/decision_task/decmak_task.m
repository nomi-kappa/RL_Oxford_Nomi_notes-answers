
%NK
addpath(genpath('/Users/nomikikoutsoubari/Documents/GitHub/RL_Oxford_Nomi_notes-answers/computational-models-course-master/Session1/decision_task/Psychtoolbox'));
savepath

% First time only
    % SetupPsychtoolbox

cd('/Users/nomikikoutsoubari/Documents/GitHub/RL_Oxford_Nomi_notes-answers/computational-models-course-master/Session1/decision_task/Psychtoolbox');

  %  SetupPsychtoolbox; % Your free trial is active for 13 more days, until
  %  Fri Oct 17 13:08:02 2025 ; to rerun: Rebuild Psychtoolbox from source (free, no license needed).

%% 
% Clear the workspace and the screen
Screen('CloseAll');
close all;
clearvars;
addpath(genpath('CogToolbox'));

%% set up the experimental variables and trial variables structures

expVariables = []; %this structure will store overall experimental variables
trialVariables = []; %this structure will contain the variables for each trial

%% get Experiment information

experimenterInitials    = upper(input('EXPERIMENTER: Please type YOUR initials (e.g. ''LH''): ','s'));
condition               = upper(input('Please enter the CONDITION ( ''D'' ''S1'' ''S2'', ''N1'', or ''N2''): ','s'));
if ~any(strcmp(condition,{'D' 'S1' 'S2' 'N1' 'N2'}))
    error('Unrecognised condition.');
end

dataFileName = fullfile('data',sprintf('condition%s_%s.mat',condition,experimenterInitials));

if exist(dataFileName,'file')&&~strcmp(condition,'D')
    error('File %s already exists - have you run this condition already? Delete the file if you need to rerun this condition.',...
        dataFileName);
end


%% load the experimental schedule, stored in 'trialVariables'
if strcmp(condition,'D')
    load('demo_schedule.mat');
elseif strcmp(condition,'S1')
    load('stable_first_aversive_schedule.mat');
elseif strcmp(condition,'S2')
    load('volatile_first_aversive_schedule.mat');
elseif strcmp(condition,'N1')
    load('stable_first_neutral_schedule.mat');
elseif strcmp(condition,'N2')
    load('volatile_first_neutral_schedule.mat');
end

% example trial structure shown below:
%trialVariables(1).magOpt1 = 34; %points available on green
%trialVariables(1).magOpt2 = 67; %points available on blue
%trialVariables(1).opt1Rewarded = 1; %1 if green rewarded, 0 if blue rewarded
%trialVariables(1).Opt1Left = 1; % 1 if green is on l.h.s. of screen
%trialVariables(1).playsound = 0; % 1 if non-aversive, 2 if aversive
%trialVariables(1).trueProbability = 0.7;

%% Setting up Psychtoolbox and the display screen
PsychDefaultSetup(2); %some default settings

% Get the screen numbers. This gives us a number for each of the screens
% attached to our computer.
screens = Screen('Screens');

% To draw we select the maximum of these numbers. So in a situation where we
% have two screens attached to our monitor we will draw to the external
% screen.
screenNumber = max(screens);

% Define black and white (white will be 1 and black 0). This is because
% in general luminace values are defined between 0 and 1 with 255 steps in
% between. All values in Psychtoolbox are defined between 0 and 1
expVariables.white = WhiteIndex(screenNumber);
expVariables.black = BlackIndex(screenNumber);

% Open an on screen window using PsychImaging and color it white
[window, windowRect] = PsychImaging('OpenWindow', screenNumber, expVariables.white);
% Get the size of the on screen window
[screenXpixels, screenYpixels] = Screen('WindowSize', window);
% Get the centre coordinate of the window
[xCentre, yCentre] = RectCenter(windowRect);

%insert these coordinates/dimensions into expVariables structure
expVariables.screenXpixels = screenXpixels;
expVariables.screenYpixels = screenYpixels;
expVariables.xCentre = xCentre;
expVariables.yCentre = yCentre;
expVariables.window  = window;

%% set up sounds
InitializePsychSound(1);

%load in neutral sounds
for i = 1:10
    wavFileName  = fullfile('sounds','chords',sprintf('chord%0.0f.wav',i));
    [expVariables.neutralSounds{i},expVariables.freq] = psychwavread(wavFileName);
    nrchannels = size(expVariables.neutralSounds{i},2); % number of channels

    % Make sure we have always 2 channels stereo output.
    % Why? Because some low-end and embedded soundcards
    % only support 2 channels, not 1 channel, and we want
    % to be robust.
    if nrchannels < 2
        expVariables.neutralSounds{i} = [expVariables.neutralSounds{i} expVariables.neutralSounds{i}];
        nrchannels = 2;
    end
end

%load in stressful sounds
for i = 1:10
    wavFileName  = fullfile('sounds','stressful',sprintf('stress%0.0f.wav',i));
    [expVariables.aversiveSounds{i},expVariables.freq] = psychwavread(wavFileName);
    nrchannels = size(expVariables.neutralSounds{i},2); % number of channels

    % Make sure we have always 2 channels stereo output.
    % Why? Because some low-end and embedded soundcards
    % only support 2 channels, not 1 channel, and we want
    % to be robust.
    if nrchannels < 2
        expVariables.aversiveSounds{i} = [expVariables.aversiveSounds{i} expVariables.aversiveSounds{i}];
        nrchannels = 2;
    end
end

try
    % Try with the 'freq'uency we wanted:
    expVariables.pahandle = PsychPortAudio('Open', [], [], 0, expVariables.freq, nrchannels);
catch
    % Failed. Retry with default frequency as suggested by device:
    fprintf('\nCould not open device at wanted playback frequency of %i Hz. Will retry with device default frequency.\n', freq);
    fprintf('Sound may sound a bit out of tune, ...\n\n');

    psychlasterror('reset');
    expVariables.pahandle = PsychPortAudio('Open', [], [], 0, [], nrchannels);
end


%% set up timing information

expVariables.itiDur      = 1; %duration of inter-trial interval, seconds
expVariables.choseDur    = 1; %duration to display chosen option, seconds
expVariables.feedbackDur = 1; %duration of post-decision feedback, seconds
expVariables.ifi         = Screen('GetFlipInterval', window); %interframe interval

%% set up keyboard information

% Define the keyboard keys that are listened for. We will be using the left
% and right arrow keys as response keys for the task and the escape key as
% a exit/reset key
expVariables.escapeKey = KbName('ESCAPE');
expVariables.leftKey = KbName('LeftArrow');
expVariables.rightKey = KbName('RightArrow');

%% run the experiment

data = []; %this structure will store subject's behaviour throughout the experiment
nTrials = length(trialVariables);

%set up structures for LikertData
likertData = [];
likertCounter = 1;
likertFrequency = 25; %how many trials between each likert rating?
for t = 1:nTrials
    [data,escaped] = run_decmak_trial_windows(trialVariables(t),data,expVariables);
    
    %this 
    if mod(t,likertFrequency)==0
        likertData.happy(likertCounter) = likert('How happy are you, from 1-10?',...
            expVariables);
        likertData.stress(likertCounter) = likert('How stressed are you, from 1-10?',...
            expVariables);
        likertData.trialnumber(likertCounter) = t;
        likertCounter = likertCounter + 1;
    end
    
    save(dataFileName,'data','trialVariables','likertData'); %saves data structure
    if escaped
        return;
    end
end

%% close up
Screen('CloseAll');

fprintf('\n============================================\n');
fprintf('This subject scored a total of %0.0f points.\n',sum(data.pointswon));
fprintf('============================================\n');

