function [behaviour,escaped] = run_decmak_trial(trialVariables, behaviour, expVariables)

% [behaviour,escaped] = run_decmak_trial(trialVariables, behaviour, expVariables)
%
% this function runs a single trial of a decision making experiment
%
% trialVariables is structure containing details of this particular trial:
%    .opt1Rewarded is vector of whether green is rewarded (1 if true)
%    .magOpt1 is points available on green
%    .magOpt2 is points available on blue
%    .Opt1Left (optional) is whether green is presented on left (random if omitted)
%    .playsound (optional) is 0 = none (default), 1 = non-aversive, 2 = aversive
%    .TrueProbability (optional) is hidden probability of green being rewarded
%    
% behaviour is a structure describing how subject has behaved thus far in experiment:
%    .opt1Chosen is vector of choices on previous trials, 1 if chose green, 0 if blue
%    .opt1Rewarded is vector of whether green was rewarded on previous trials
%    .reactionTime is vector of reaction times on previous trials
%    .chosenOptionRewarded is vector of whether chosen option was rewarded on previous trials
%    .opt1Rewarded is vector of whether green was rewarded on previous trials (1 if true) 
%    .magOpt1 is vector of points available on green on previous trials
%    .magOpt2 is vector of points available on blue on previous trials
%    .pointswon is how many points were won on each previous trial
%    .Opt1left is whether green was presented on left
%    .TrueProbability is underlying vector of true probabilities
%    .playsound is whether sound was played (0 = none, 1 = non-aversive, 2 = aversive)
%    .reactionTime is reactionTime in ms
%    .isStableBlock (optional) is whether current block is stable
%    --> it is returned as an output with current trial behaviour appended
%    --> for first trial, behaviour can be empty variable (will be created)
%
% expVariables is structure containing variables of the entire experiment
%    (see decmak_task.m for details)
%
% escaped is normally 0, but will be set to 1 if the subject presses Escape key
%
% Laurence Hunt, September 2017

%% get experimental variables

screenXpixels   = expVariables.screenXpixels;
screenYpixels   = expVariables.screenYpixels;
xCentre         = expVariables.xCentre;
yCentre         = expVariables.yCentre;
white           = expVariables.white;
black           = expVariables.black;
window          = expVariables.window;
waitframes      = 1;

escapeKey       = expVariables.escapeKey;
leftKey         = expVariables.leftKey;
rightKey        = expVariables.rightKey;

ifi             = expVariables.ifi; %interframe interval
itiFrames       = ceil(expVariables.itiDur/ifi); %duration of inter-trial interval, in frames
choseFrames     = ceil(expVariables.choseDur/ifi); %duration of inter-trial interval, in frames
feedbackFrames  = ceil(expVariables.feedbackDur/ifi); %duration of inter-trial interval, in frames

pointsScaleFactor = 8; %can adjust to alter speed at which pointsbar grows.

%calculate total points won thus far
if isempty(behaviour) %first trial
    totalPointsWon = 0;
else
    totalPointsWon = sum(behaviour.pointswon);
end

%% get trial variables
opt1Rewarded    = trialVariables.opt1Rewarded;
magOpt1         = trialVariables.magOpt1; 
magOpt2         = trialVariables.magOpt2; 
try Opt1left    = trialVariables.Opt1left; catch, Opt1left = rand>0.5; end
try TrueProb    = trialVariables.TrueProbability; catch, TrueProb = nan; end
try playsound   = trialVariables.playsound; catch, playsound = 0; end
%determine the current trial number, from past behaviour
if isempty(behaviour) %behaviour will be empty if this is the first trial
    trialnumber = 1;
else
    trialnumber = length(behaviour.opt1Chosen)+1;
end

if Opt1left
    Opt1X = screenXpixels * 0.25;
    Opt2X = screenXpixels * 0.75;
else
    Opt1X = screenXpixels * 0.75;
    Opt2X = screenXpixels * 0.25;
end
textXoffset = screenXpixels *0.017;
textYoffset = screenYpixels *0.007;
pointsXoffset = screenXpixels *0.1;
pointsYoffset = screenYpixels *0.47;
pointsbarXoffset = screenXpixels *0.1;
pointsbarYoffset = screenYpixels *0.825;


% Set the colors to Green and Blue
opt1Color = [0 1 0];
opt2Color = [0.5 0 1];
chosenColor=[0.5 0.5 0.5];
rewardedColor=[0.5 0.5 0];
pointsColor = [0.7529    0.7529    0.7529];

% Make a base Rect of 200 by 400 pixels
baseRect = [0 0 275 400];
opt1Rect = CenterRectOnPointd(baseRect, Opt1X, yCentre);
opt2Rect = CenterRectOnPointd(baseRect, Opt2X, yCentre);

miniRect = [0 0 100 50];
opt1miniRect = CenterRectOnPointd(miniRect, Opt1X, yCentre);
opt2miniRect = CenterRectOnPointd(miniRect, Opt2X, yCentre);

%set up rewardedRect, highlighting the rewarded option
if opt1Rewarded == 1
    rewardedRect = CenterRectOnPointd(baseRect*1.35, Opt1X, yCentre);
else
    rewardedRect = CenterRectOnPointd(baseRect*1.35, Opt2X, yCentre);
end

%set up pointsRect, which shows the current points total
if isfield(behaviour,'pointswon');
    pbwidth = sum(behaviour.pointswon)/pointsScaleFactor+10;
else
    pbwidth = 10;
end
pointsRect = [pointsbarXoffset pointsbarYoffset ...
                pointsbarXoffset+pbwidth...
                pointsbarYoffset*1.1];

Screen('TextSize', window, 40);

%% run trial

% 1. run the inter-trial interval
Screen('DrawDots', window, [xCentre; yCentre], 10, black, [], 2);
DrawFormattedText(window, sprintf('Your score: %0.0f', totalPointsWon), xCentre-pointsXoffset, ...
    yCentre+pointsYoffset, [0 0 0]);
Screen('FillRect', window, pointsColor, pointsRect);
vbl = Screen('Flip', window);

% 2. present the two options on left and right
Screen('DrawDots', window, [xCentre; yCentre], 10, black, [], 2);
Screen('FillRect', window, opt1Color, opt1Rect);
Screen('FillRect', window, opt2Color, opt2Rect);
Screen('FillRect', window, [1 1 1]*black, opt1miniRect);
Screen('FillRect', window, [1 1 1]*black, opt2miniRect);
Screen('FillRect', window, pointsColor, pointsRect);
DrawFormattedText(window, num2str(magOpt1), Opt1X-textXoffset, ...
    yCentre+textYoffset, [1 1 1]);
DrawFormattedText(window, num2str(magOpt2), Opt2X-textXoffset, ...
    yCentre+textYoffset, [1 1 1]);
DrawFormattedText(window, sprintf('Your score: %0.0f', totalPointsWon), xCentre-pointsXoffset, ...
    yCentre+pointsYoffset, [0 0 0]);
vbl = Screen('Flip', window, vbl + (itiFrames - 0.5) * ifi);

%3. wait for response
respToBeMade = true;
tStart = GetSecs;
while respToBeMade == true
    [keyIsDown,secs, keyCode] = KbCheck;
    if keyCode(escapeKey)
        ShowCursor;
        sca;
        escaped = 1;
        return
    elseif keyCode(leftKey)
        response = 1;
        respToBeMade = false;
    elseif keyCode(rightKey)
        response = 2;
        respToBeMade = false;
    end
end
tEnd = GetSecs;
reactionTime = tEnd - tStart;

%4. highlight chosen option
if (response==1&&Opt1left==1)||(response==2&&Opt1left==0);
    chosenRect = CenterRectOnPointd(baseRect*1.2, Opt1X, yCentre);
    opt1Chosen = 1; %chose option 1 (green)
else
    chosenRect = CenterRectOnPointd(baseRect*1.2, Opt2X, yCentre);
    opt1Chosen = 0; %chose option 2 (blue)
end
Screen('DrawDots', window, [xCentre; yCentre], 10, black, [], 2);
Screen('FillRect', window, chosenColor, chosenRect);
Screen('FillRect', window, opt1Color, opt1Rect);
Screen('FillRect', window, opt2Color, opt2Rect);
Screen('FillRect', window, [1 1 1]*black, opt1miniRect);
Screen('FillRect', window, [1 1 1]*black, opt2miniRect);
Screen('FillRect', window, pointsColor, pointsRect);
DrawFormattedText(window, num2str(magOpt1), Opt1X-textXoffset, ...
    yCentre+textYoffset, [1 1 1]);
DrawFormattedText(window, num2str(magOpt2), Opt2X-textXoffset, ...
    yCentre+textYoffset, [1 1 1]);
DrawFormattedText(window, sprintf('Your score: %0.0f', totalPointsWon), xCentre-pointsXoffset, ...
    yCentre+pointsYoffset, [0 0 0]);
vbl = Screen('Flip', window);

%calculate if opt1Chosen was rewarded
if (opt1Chosen==1)&&(opt1Rewarded==1)
    chosenOptionRewarded = 1;
    pointswon = magOpt1;
elseif (opt1Chosen==0)&&(opt1Rewarded==0)
    chosenOptionRewarded = 1;
    pointswon = magOpt2;
else
    chosenOptionRewarded = 0;
    pointswon = 0;
end

pbwidth = pbwidth + pointswon/pointsScaleFactor;
pointsRect = [pointsbarXoffset pointsbarYoffset ...
                pointsbarXoffset+pbwidth...
                pointsbarYoffset*1.1];
            
%5. reveal rewarded option
Screen('DrawDots', window, [xCentre; yCentre], 10, black, [], 2);
Screen('FillRect', window, rewardedColor, rewardedRect);
Screen('FillRect', window, chosenColor, chosenRect);
Screen('FillRect', window, opt1Color, opt1Rect);
Screen('FillRect', window, opt2Color, opt2Rect);
Screen('FillRect', window, [1 1 1]*black, opt1miniRect);
Screen('FillRect', window, [1 1 1]*black, opt2miniRect);
Screen('FillRect', window, pointsColor, pointsRect);
DrawFormattedText(window, num2str(magOpt1), Opt1X-textXoffset, ...
    yCentre+textYoffset, [1 1 1]);
DrawFormattedText(window, num2str(magOpt2), Opt2X-textXoffset, ...
    yCentre+textYoffset, [1 1 1]);
DrawFormattedText(window, sprintf('Your score: %0.0f', totalPointsWon), xCentre-pointsXoffset, ...
    yCentre+pointsYoffset, [0 0 0]);
vbl = Screen('Flip', window, vbl + (choseFrames - 0.5) * ifi);
if playsound==1 %play randomly chosen neutral sound
    soundToPlay = ceil(rand*10);
    PsychPortAudio('FillBuffer', expVariables.pahandle, expVariables.neutralSounds{soundToPlay}');
    t1 = PsychPortAudio('Start', expVariables.pahandle, 1, 0, 1);
elseif playsound==2 %play randomly chosen aversive sound
    soundToPlay = ceil(rand*10);
    PsychPortAudio('FillBuffer', expVariables.pahandle, expVariables.aversiveSounds{soundToPlay}');
    t1 = PsychPortAudio('Start', expVariables.pahandle, 1, 0, 1);
end

%6. flip screen back to fixation point
Screen('DrawDots', window, [xCentre; yCentre], 10, black, [], 2);
Screen('FillRect', window, pointsColor, pointsRect);
vbl = Screen('Flip', window, vbl + (feedbackFrames - 0.5) * ifi);

%% store subject behaviour
behaviour.opt1Chosen(trialnumber) = opt1Chosen;
behaviour.magOpt1(trialnumber) = magOpt1;
behaviour.magOpt2(trialnumber) = magOpt2;
behaviour.opt1Rewarded(trialnumber) = opt1Rewarded;
behaviour.reactionTime(trialnumber) = reactionTime;
behaviour.chosenOptionRewarded(trialnumber) = chosenOptionRewarded;
behaviour.opt1Rewarded(trialnumber) = opt1Rewarded;
try 
    behaviour.isStableBlock(trialnumber) = trialVariables.isStableBlock; 
catch
    behaviour.isStableBlock(trialnumber) = nan; 
end
behaviour.pointswon(trialnumber) = pointswon;
behaviour.Opt1left(trialnumber) = Opt1left;
try
    behaviour.trueProbability(trialnumber) = trialVariables.trueProbability;
catch
    behaviour.trueProbability(trialnumber) = nan;
end
behaviour.playsound(trialnumber) = playsound;
escaped = 0;

%% make sure behaviour contains column vectors, not row vectors, to keep Nils happy.
fields = fieldnames(behaviour);
for i = 1:numel(fields)
    if ~iscolumn(behaviour.(fields{i}))
        behaviour.(fields{i}) = behaviour.(fields{i})';
    end
end
