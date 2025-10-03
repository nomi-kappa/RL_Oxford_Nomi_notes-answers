function EyeLink_BufferedFixUpdateEvents(screenNumber)
% A simple EyeLink gaze-contingent demo showing how to retrieve online events from a buffer.
% In each trial an image is presented with a red gaze-contingent dot overlaid on top.
% The dot's location is based on the average x y coordinates of fixations updated online every 50ms via a FIXUPDATE event.
% See EyeLink Programmers Guide manual > Experiment Templates Overview > Control > Fixation Update Events
% Each trial ends when the space bar is pressed.
%
% FIXUPDATE events send updates about a current fixation at regular intervals. By default an interval of 50ms is used. 
% The first update is sent one update interval after the start of the fixation, and the last is sent at
% the end of the fixation. This demo uses FIXUPDATE events to get the averaged gaze x y position across each fixation interval. 
%
% Usage:
% Eyelink_BufferedFixUpdateEvents(screenNumber)
%
% screenNumber is an optional parameter which can be used to pass a specific value to PsychImaging('OpenWindow', ...)
% If screenNumber is not specified, or if isempty(screenNumber) then the default:
% screenNumber = max(Screen('Screens'));
% will be used.
%
% This demo uses the 'GetNextDataType'/'GetFloatData' function pair that allows access to the following buffered samples and events
% (See EyeLink Programmers Guide manual > Data Structures > FEVENT):
%
% STARTBLINK 3 (the start of a blink)
% ENDBLINK 4 (the end of a blink)
% STARTSACC 5 (the start of a saccade)
% ENDSACC 6 (the end of a saccade)
% STARTFIX 7 (the start of a fixation)
% ENDFIX 8 (the end of a fixation)
% FIXUPDATE 9 (a fixation update during a fixation)
% SAMPLE_TYPE 200 (a sample)
% MISSING_DATA -32768 (missing data)
%
% Use buffered data if you need to:
% a) grab every single consecutive sample online
% b) grab event data (e.g. fixation/saccade/blink events) online
%
% Note that some buffered event data take some time to be available online due to the time involved
% in calculating velocity/acceleration. If you need to retrieve online gaze
% position as fast as possible and/or you don't need to get all subsequent samples or other
% events, then use the Eyelink('NewFloatSampleAvailable') / Eyelink('NewestFloatSample') function pair,
% as illustrated in the GCfastSamples.m example.
% ---------------------------------------------------------------------------------------------
%
% Events structure and fields available via the 'GetNextDataType'/'GetFloatData' function pair:
% STARTBLINK, STARTSACC, STARTFIX:
%       type (number assigned to event - STARTBLINK=3, STARTSACC=5, STARTFIX=7)
%       eye (0=left eye, 1=right eye)
%       sttime (event start time)
%
% ENDBLINK:
%       type (number assigned to event - ENDBLINK=4)
%       eye (0=left eye, 1=right eye)
%       sttime (event start time)
%       entime (event end time)
%
% ENDSACC:
%       type (number assigned to event - ENDSACC=6)
%       eye (0=left eye, 1=right eye)
%       sttime (event start time)
%       entime (event end time)       
%       gstx (Saccade start x gaze position)
%       gsty (Saccade start y gaze position)
%       genx (Saccade end x gaze position)
%       geny (Saccade end y gaze position)
%       supd_x (Saccade start x 'pixel per degree' value)
%       supd_y (Saccade start y 'pixel per degree' value)
%       eupd_x (Saccade end x 'pixel per degree' value)
%       eupd_y (Saccade end y 'pixel per degree' value)
%
% FIXUPDATE, ENDFIX:
%       type (number assigned to event - FIXUPDATE=9, ENDFIX=8)
%       eye (0=left eye, 1=right eye)
%       sttime (event start time)
%       entime (event end time)  
%       gavx (average gaze x position during fixation)
%       gavy (average gaze y position during fixation)
%       ava (average pupil size)
%       supd_x (Fixation start x 'pixel per degree' value)
%       supd_y (Fixation start y 'pixel per degree' value)
%       eupd_x (Fixation end x 'pixel per degree' value)
%       eupd_y (Fixation end y 'pixel per degree' value)
%
% SAMPLE_TYPE
%       time (sample time)
%       type (SAMPLE=200)
%       pa ([lef eye pupil size, right eye pupil size])
%       gx ([left gaze x, right gaze x])
%       gy ([left gaze y, right gaze y])
%       rx (x 'pixel per degree' value)
%       ry (y 'pixel per degree' value)
%       buttons (button state and changes)
%       hdata (contains a list of 8 fields. Only the first 4 values are important: 
%             [uncalibrated target sticker x, uncalibrated target sticker y, target sticker distance in mm, target flags)

% Bring the Command Window to the front if it is already open
if ~IsOctave; commandwindow; end

PsychDefaultSetup(2);

% Use default screenNumber if none specified
if (nargin < 1)
    screenNumber = [];
end
try
    %% STEP 1: INITIALIZE EYELINK CONNECTION; OPEN EDF FILE; GET EYELINK TRACKER VERSION
    
    % Initialize EyeLink connection (dummymode = 0) or run in "Dummy Mode" without an EyeLink connection (dummymode = 1);
    dummymode = 0;
    EyelinkInit(dummymode); % Initialize EyeLink connection
    status = Eyelink('IsConnected');
    if status < 1 % If EyeLink not connected
        dummymode = 1; 
    end
    
    % Open dialog box for EyeLink Data file name entry. File name up to 8 characters
    prompt = {'Enter EDF file name (up to 8 characters)'};
    dlg_title = 'Create EDF file';
    def = {'demo'}; % Create a default edf file name
    answer = inputdlg(prompt, dlg_title, 1, def); % Prompt for new EDF file name    
    % Print some text in Matlab's Command Window if a file name has not been entered
    if  isempty(answer)
        fprintf('Session cancelled by user\n')
        cleanup; % Abort experiment (see cleanup function below)
        return
    end    
    edfFile = answer{1}; % Save file name to a variable    
    % Print some text in Matlab's Command Window if file name is longer than 8 characters
    if length(edfFile) > 8
        fprintf('Filename needs to be no more than 8 characters long (letters, numbers and underscores only)\n');
        cleanup; % Abort experiment (see cleanup function below)
        return
    end
    
    % Open an EDF file and name it
    failOpen = Eyelink('OpenFile', edfFile);
    if failOpen ~= 0 % Abort if it fails to open
        fprintf('Cannot create EDF file %s', edfFile); % Print some text in Matlab's Command Window
        cleanup; %see cleanup function below
        return
    end
    
    % Get EyeLink tracker and software version
    % <ver> returns 0 if not connected
    % <versionstring> returns 'EYELINK I', 'EYELINK II x.xx', 'EYELINK CL x.xx' where 'x.xx' is the software version
    ELsoftwareVersion = 0; % Default EyeLink version in dummy mode
    [ver, versionstring] = Eyelink('GetTrackerVersion');
    if dummymode == 0 % If connected to EyeLink
        % Extract software version number. 
        [~, vnumcell] = regexp(versionstring,'.*?(\d)\.\d*?','Match','Tokens'); % Extract EL version before decimal point
        ELsoftwareVersion = str2double(vnumcell{1}{1}); % Returns 1 for EyeLink I, 2 for EyeLink II, 3/4 for EyeLink 1K, 5 for EyeLink 1KPlus, 6 for Portable Duo         
        % Print some text in Matlab's Command Window
        fprintf('Running experiment on %s version %d\n', versionstring, ver );
    end
    % Add a line of text in the EDF file to identify the current experimemt name and session. This is optional.
    % If your text starts with "RECORDED BY " it will be available in DataViewer's Inspector window by clicking
    % the EDF session node in the top panel and looking for the "Recorded By:" field in the bottom panel of the Inspector.
    preambleText = sprintf('RECORDED BY Psychtoolbox demo %s session name: %s', mfilename, edfFile);
    Eyelink('Command', 'add_file_preamble_text "%s"', preambleText);
    
    
    %% STEP 2: SELECT AVAILABLE SAMPLE/EVENT DATA
    % See EyeLinkProgrammers Guide manual > Useful EyeLink Commands > File Data Control & Link Data Control
    
    % Select which events are saved in the EDF file. Include everything just in case
    Eyelink('Command', 'file_event_filter = LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,BUTTON,INPUT');
    % Select which events are available online for gaze-contingent experiments. Include everything just in case
    Eyelink('Command', 'link_event_filter = LEFT,RIGHT,FIXATION,SACCADE,BLINK,BUTTON,FIXUPDATE,INPUT');
    % Select which sample data is saved in EDF file or available online. Include everything just in case
    if ELsoftwareVersion > 3  % Check tracker version and include 'HTARGET' to save head target sticker data for supported eye trackers
        Eyelink('Command', 'file_sample_data  = LEFT,RIGHT,GAZE,HREF,RAW,AREA,HTARGET,GAZERES,BUTTON,STATUS,INPUT');
        Eyelink('Command', 'link_sample_data  = LEFT,RIGHT,GAZE,GAZERES,AREA,HTARGET,STATUS,INPUT');
    else
        Eyelink('Command', 'file_sample_data  = LEFT,RIGHT,GAZE,HREF,RAW,AREA,GAZERES,BUTTON,STATUS,INPUT');
        Eyelink('Command', 'link_sample_data  = LEFT,RIGHT,GAZE,GAZERES,AREA,STATUS,INPUT');
    end
    
    
    %% STEP 3: OPEN GRAPHICS WINDOW
    
    % Open experiment graphics on the specified screen
    if isempty(screenNumber)
        screenNumber = max(Screen('Screens')); % Use default screen if none specified
    end

    window = PsychImaging('OpenWindow', screenNumber, GrayIndex(screenNumber)); % Open graphics window
    Screen('Flip', window);
    
    % Get max color value for rescaling  to RGB for Host PC & Data Viewer integration
    colorMaxVal = Screen('ColorRange', window);
    % Return width and height of the graphics window/screen in pixels
    [width, height] = Screen('WindowSize', window);
    
    
    %% STEP 4: SET CALIBRATION SCREEN COLOURS; PROVIDE WINDOW SIZE TO EYELINK HOST & DATAVIEWER; SET CALIBRATION PARAMETERS; CALIBRATE
    
    % Provide EyeLink with some defaults, which are returned in the structure "el".
    el = EyelinkInitDefaults(window);
    % set calibration/validation/drift-check(or drift-correct) size as well as background and target colors. 
    % It is important that this background colour is similar to that of the stimuli to prevent large luminance-based 
    % pupil size changes (which can cause a drift in the eye movement data)
    el.calibrationtargetsize = 3;% Outer target size as percentage of the screen
    el.calibrationtargetwidth = 0.7;% Inner target size as percentage of the screen
    el.backgroundcolour = repmat(GrayIndex(screenNumber),1,3);
    el.calibrationtargetcolour = repmat(BlackIndex(screenNumber),1,3);
    % set "Camera Setup" instructions text colour so it is different from background colour
    el.msgfontcolour = repmat(BlackIndex(screenNumber),1,3);

    % Initialize PsychSound for calibration/validation audio feedback
    % EyeLink Toolbox now supports PsychPortAudio integration and interop
    % with legacy Snd() wrapping. Below we open the default audio device in
    % output mode as master, create a slave device, and pass the device
    % handle to el.ppa_pahandle.
    % el.ppa_handle supports passing either standard mode handle, or as
    % below one opened as a slave device. When el.ppa_handle is empty, for
    % legacy support EyelinkUpdateDefaults() will open the default device
    % and use that with Snd() interop, and close the device handle when
    % calling Eyelink('Shutdown') at the end of the script.
    InitializePsychSound();
    pamaster = PsychPortAudio('Open', [], 8+1);
    PsychPortAudio('Start', pamaster);
    pahandle = PsychPortAudio('OpenSlave', pamaster, 1);
    el.ppa_pahandle = pahandle;

    % You must call this function to apply the changes made to the el structure above
    EyelinkUpdateDefaults(el);
    
    % Set display coordinates for EyeLink data by entering left, top, right and bottom coordinates in screen pixels
    Eyelink('Command','screen_pixel_coords = %ld %ld %ld %ld', 0, 0, width-1, height-1);
    % Write DISPLAY_COORDS message to EDF file: sets display coordinates in DataViewer
    % See DataViewer manual section: Protocol for EyeLink Data to Viewer Integration > Pre-trial Message Commands
    Eyelink('Message', 'DISPLAY_COORDS %ld %ld %ld %ld', 0, 0, width-1, height-1);
       
    % Set number of calibration/validation dots and spread: horizontal-only(H) or horizontal-vertical(HV) as H3, HV3, HV5, HV9 or HV13
    Eyelink('Command', 'calibration_type = HV9'); % horizontal-vertical 9-points
    % Allow a supported EyeLink Host PC button box to accept calibration or drift-check/correction targets via button 5
    Eyelink('Command', 'button_function 5 "accept_target_fixation"');
    % Hide mouse cursor
    HideCursor(window);
    % Suppress keypress output to command window.
    ListenChar(-1);
    Eyelink('Command', 'clear_screen 0'); % Clear Host PC display from any previus drawing
    % Put EyeLink Host PC in Camera Setup mode for participant setup/calibration
    EyelinkDoTrackerSetup(el);
    
    
    %% STEP 5: TRIAL LOOP.
    
    spaceBar = KbName('space');% Identify keyboard key code for space bar to end each trial later on
    imgList = {'img1.jpg' 'img2.jpg'};% Provide image list for 2 trials
    for i = 1:length(imgList)
        Sstim = 0; % Reset initial stimulus marker
        
        % STEP 5.1: PREBUILD STIMULUS (GREY BACKGROUND + IMAGE + TEXT)
        
        % Prepare grey background on backbuffer
        Screen('FillRect', window, el.backgroundcolour);
        % Use 'drawBuffer' to copy unprocessed backbuffer images without additional processing. Prevents image size info issues on Retina displays
        backgroundArray = Screen('GetImage', window, [], 'drawBuffer'); % Copy unprocessed backbuffer
        backgroundTexture = Screen('MakeTexture', window, backgroundArray); % Convert background to texture so it is ready for drawing later on        
        % Prepare image on backbuffer
        imgName = char(imgList(i)); % Get image file name for current trial
        imgInfo = imfinfo(imgName); % Get image file info
        imgData = imread(imgName); % Read image from file
        imgTexture = Screen('MakeTexture',window, imgData); % Convert image file to texture
        Screen('DrawTexture', window, imgTexture); % Prepare image texture on backbuffer        
        % Prepare text on backbuffer
        Screen('TextSize', window, 30); % Specify text size
        Screen('DrawText', window, 'Press space bar to end trial', 5, height-35, 0); % Prepare text on backbuffer        
        % Save complete backbuffer as trial*.bmp to be used as stimulus and as Host PC & DataViewer backdrop
        stimName = ['trial' num2str(i) '.bmp']; % Prepare stimulus file name
        stimArray = Screen('GetImage', window, [], 'drawBuffer'); % Copy backbuffer to be used as stimulus
        imwrite(stimArray, stimName); % Save .bmp stimulus file in experment folder        
        % Convert stimulus to texture so it is ready for drawing later on
        stimInfo = imfinfo(stimName); % Get stimulus info
        stimTexture = Screen('MakeTexture', window, stimArray); % Convert to texture
                
        % STEP 5.2: START TRIAL; SHOW TRIAL INFO ON HOST PC; SHOW BACKDROP IMAGE AND/OR DRAW FEEDBACK GRAPHICS ON HOST PC; DRIFT-CHECK/CORRECTION
        
        % Write TRIALID message to EDF file: marks the start of a trial for DataViewer
        % See DataViewer manual section: Protocol for EyeLink Data to Viewer Integration > Defining the Start and End of a Trial
        Eyelink('Message', 'TRIALID %d', i);
        
        % Write !V CLEAR message to EDF file: creates blank backdrop for DataViewer
        % See DataViewer manual section: Protocol for EyeLink Data to Viewer Integration > Simple Drawing
        Eyelink('Message', '!V CLEAR %d %d %d', round(el.backgroundcolour(1)/colorMaxVal*255), round(el.backgroundcolour(2)/colorMaxVal*255), round(el.backgroundcolour(3)/colorMaxVal*255));
        
        % Supply the trial number as a line of text on Host PC screen
        Eyelink('Command', 'record_status_message "TRIAL %d/%d"', i, length(imgList));
        
        % Draw graphics on the EyeLink Host PC display. See COMMANDS.INI in the Host PC's exe folder for a list of commands
        Eyelink('SetOfflineMode'); % Put tracker in idle/offline mode before drawing Host PC graphics and before recording        
        Eyelink('Command', 'clear_screen 0'); % Clear Host PC display from any previus drawing
        % Optional: Send an image to the Host PC to be displayed as the backdrop image over which 
        % the gaze-cursor is overlayed during trial recordings.
        % See Eyelink('ImageTransfer?') for information about supported syntax and compatible image formats.
        % Below, we use the new option to pass image data from imread() as the imageArray parameter, which
        % enables the use of many image formats.
        % [status] = Eyelink('ImageTransfer', imageArray, xs, ys, width, height, xd, yd, options);
        % xs, ys: top-left corner of the region to be transferred within the source image
        % width, height: size of region to be transferred within the source image (note, values of 0 will include the entire width/height)
        % xd, yd: location (top-left) where image region to be transferred will be presented on the Host PC
        % This image transfer function works for non-resized image presentation only. If you need to resize images and use this function please resize
        % the original image files beforehand
        transferStatus = Eyelink('ImageTransfer', stimArray, 0, 0, 0, 0, 0, 0);
        if dummymode == 0 && transferStatus ~= 0 % If connected to EyeLink and image transfer fails
            fprintf('Image transfer Failed\n'); % Print some text in Matlab's Command Window
        end
        
        % Optional: draw feedback box and lines on Host PC interface instead of (or on top of) backdrop image.
        % See section 25.7 'Drawing Commands' in the EyeLink Programmers Guide manual
        Eyelink('Command', 'draw_box %d %d %d %d 15', round(width/2-imgInfo.Width/2), round(height/2-imgInfo.Height/2), round(width/2+imgInfo.Width/2), round(height/2+imgInfo.Height/2));
        
        % Perform a drift check/correction.
        % Optionally provide x y target location, otherwise target is presented on screen centre
        EyelinkDoDriftCorrection(el, round(width/2), round(height/2));
                
        %STEP 5.3: START RECORDING
        
        % Put tracker in idle/offline mode before recording. Eyelink('SetOfflineMode') is recommended 
        % however if Eyelink('Command', 'set_idle_mode') is used allow 50ms before recording as shown in the commented code:        
        % Eyelink('Command', 'set_idle_mode');% Put tracker in idle/offline mode before recording
        % WaitSecs(0.05); % Allow some time for transition           
        Eyelink('SetOfflineMode');% Put tracker in idle/offline mode before recording
        Eyelink('StartRecording'); % Start tracker recording
        WaitSecs(0.1); % Allow some time to record a few samples before presenting first stimulus
                
        % STEP 5.4: PRESENT STIMULUS; CREATE DATAVIEWER BACKDROP AND INTEREST AREA
        
        % Check which eye is available for gaze-contingent drawing. Returns 0 (left), 1 (right) or 2 (binocular)
        eyeUsed = Eyelink('EyeAvailable');
        % Get events from right eye if binocular
        if eyeUsed == 2
            eyeUsed = 1;
        end
        
        bufferStart = GetSecs; % Start a ~100ms counter
        while 1 % loop until error or space bar is press            
            % Check that eye tracker is  still recording. Otherwise close and transfer copy of EDF file to Display PC
            err = Eyelink('CheckRecording');
            if(err ~= 0)
                fprintf('EyeLink Recording stopped!\n');
                % Transfer a copy of the EDF file to Display PC
                Eyelink('SetOfflineMode'); % Put tracker in idle/offline mode
                Eyelink('CloseFile'); % Close EDF file on Host PC
                Eyelink('Command', 'clear_screen 0'); % Clear trial image on Host PC at the end of the experiment
                WaitSecs(0.1); % Allow some time for screen drawing
                % Transfer a copy of the EDF file to Display PC
                transferFile; % See transferFile function below
                cleanup; % Abort experiment (see cleanup function below)
                return
            end
            
            % Run the 'GetNextDataType'/'GetFloatData' function pair in a loop for ~100ms before presenting initial stimulus.
            % This will clear old data from the buffer and allow access to the most recent online FIXUPDATE event.
            if GetSecs - bufferStart > 0.1 && Sstim == 0 % If 100ms have elapsed and initial trial image has not yet been presented...                
                % Present initial trial image without gaze-contingent dot
                Screen('DrawTexture', window, stimTexture); % Prepare stimulus texture on backbuffer
                Screen('Flip', window); % Present stimulus
                % Write message to EDF file to mark the start time of stimulus presentation.
                Eyelink('Message', 'STIM_ONSET');                
                % Write !V IMGLOAD message to EDF file: creates backdrop image for DataViewer
                % See DataViewer manual section: Protocol for EyeLink Data to Viewer Integration > Image Commands
                Eyelink('Message', '!V IMGLOAD CENTER %s %d %d', stimName, width/2, height/2);               
                % Write !V IAREA message to EDF file: creates interest area around image in DataViewer
                % See DataViewer manual section: Protocol for EyeLink Data to Viewer Integration > Interest Area Commands
                Eyelink('Message', '!V IAREA RECTANGLE %d %d %d %d %d %s', 1, round(width/2-imgInfo.Width/2), round(height/2-imgInfo.Height/2), round(width/2+imgInfo.Width/2), round(height/2+imgInfo.Height/2),'IMAGE_IA');

                % Return the current EDF time (in seconds) to make sure we use only online events that started after stimulus presentation
                StimEDFtime = (Eyelink('TrackerTime'))*1000; % Multiply by 1000 to convert to milliseconds                               
                Sstim = 1; % Initial stimulus has been drawn
            end
            
            % STEP 5.5: DRAW GAZE-CONTINGENT DOT; WAIT FOR KEYPRESS; SHOW BLANK SCREEN; STOP RECORDING
            
            % Get next data item (sample or event) from link buffer.
            % This is equivalent to EyeLink_get_next_data() in C API. See EyeLink Programmers Guide manual > Message and Command Sending/Receiving > Functions
            evtype = Eyelink('GetNextDataType');
            
            % Read item type returned by getnextdatatype. Wait for a fixation update (FIXUPDATE) event
            % 'GetFloatData' is equivalent to eyelink_get_float_data() in C API. See EyeLink Programmers Guide manual > Function Lists > Message and Command Sending/Receiving > Functions
            if evtype == el.FIXUPDATE % if fixation update (FIXUPDATE) event is returned
                evt = Eyelink('GetFloatData', evtype); % access the FIXUPDATE event structure                
                if Sstim == 1 % Only draw gaze-contingent dot after initial trial image is presented (i.e. after looping for ~100ms)                   
                    % evt.eye returns the eye (0=left, 1=right) for this event. When recording binocular data both left and right eye events are available
                    % Use event only if it is from the expected eye and start of event (evt.sttime) occurred after trial image onset
                    if evt.eye == eyeUsed && evt.sttime > StimEDFtime                                                                                             
                        % Save event fields as variables. See EyeLink Programmers Guide manual > Data Structures > FEVENT
                        x_avg = evt.gavx; % Fixation update average x position
                        y_avg = evt.gavy; % Fixation update average y position
                        
                        % The following event properties are not used in this demo but are available from the FIXUPDATE event:
                        %       type (number assigned to event - FIXUPDATE=9, ENDFIX=8)
                        %       eye (0=left eye, 1=right eye)
                        %       sttime (FIXUPDATE interval start time)
                        %       entime (FIXUPDATE interval end time)  
                        %       gavx (average gaze x position during FIXUPDATE interval)
                        %       gavy (average gaze y position during FIXUPDATE interval)
                        %       ava (average pupil size)
                        %       supd_x (Fixation start x 'pixel per degree' value)
                        %       supd_y (Fixation start y 'pixel per degree' value)
                        %       eupd_x (Fixation end x 'pixel per degree' value)
                        %       eupd_y (Fixation end y 'pixel per degree' value)
                        
                        % Draw gaze-contingent dot
                        Screen('DrawTexture', window, stimTexture); % Prepare stimulus texture on backbuffer
                        % Superimpose a red gaze-contingent dot using FIXUPDATE avg x y gaze coordinates
                        Screen('FillOval', window, [255,0,0], [x_avg-20, y_avg-20, x_avg+20, y_avg+20]);
                        Screen('Flip', window); % Present stimulus
                        % Write !V IMGLOAD message to EDF file: redraw backdrop image for DataViewer before drawing the new target location on top
                        % See DataViewer manual section: Protocol for EyeLink Data to Viewer Integration > Image Commands
                        Eyelink('Message', '!V IMGLOAD CENTER %s %d %d', stimName, width/2, height/2);
                        % Write !V FIXPOINT message to EDF file: draws the new target location on top of the trial image in DataViewer
                        % See DataViewer manual section: Protocol for EyeLink Data to Viewer Integration > Simple Drawing Commands
                        Eyelink('Message', '!V FIXPOINT %d %d %d %d %d %d %d %d %d %d', 255, 0, 0, 255, 0, 0, round(x_avg), round(y_avg), 40, 40, x_avg, y_avg);                      
                    end
                end
            end            
            % End trial if space bar is pressed
            [~, ~, keyCode] = KbCheck;
            if keyCode(spaceBar)
                % Write message to EDF file to mark the space bar press time
                Eyelink('Message', 'KEY_PRESSED');
                break;
            end                       
        end % End of while loop
        
        % Draw blank screen at end of trial
        Screen('DrawTexture', window, backgroundTexture); % Prepare background texture on backbuffer
        Screen('Flip', window); % Present blank screen
        % Write message to EDF file to mark time when blank screen is presented
        Eyelink('Message', 'BLANK_SCREEN');
        % Write !V CLEAR message to EDF file: creates blank backdrop for DataViewer
        % See DataViewer manual section: Protocol for EyeLink Data to Viewer Integration > Simple Drawing
        Eyelink('Message', '!V CLEAR %d %d %d', round(el.backgroundcolour(1)/colorMaxVal*255), round(el.backgroundcolour(2)/colorMaxVal*255), round(el.backgroundcolour(3)/colorMaxVal*255));
        
        % Stop recording eye movements at the end of each trial
        WaitSecs(0.1); % Add 100 msec of data to catch final events before stopping
        Eyelink('StopRecording'); % Stop tracker recording
               
        % STEP 5.6: CREATE VARIABLES FOR DATAVIEWER; END TRIAL
        
        % Write !V TRIAL_VAR messages to EDF file: creates trial variables in DataViewer
        % See DataViewer manual section: Protocol for EyeLink Data to Viewer Integration > Trial Message Commands
        Eyelink('Message', '!V TRIAL_VAR iteration %d', i); % Trial iteration
        Eyelink('Message', '!V TRIAL_VAR image %s', imgName); % Image name       
        % Write TRIAL_RESULT message to EDF file: marks the end of a trial for DataViewer
        % See DataViewer manual section: Protocol for EyeLink Data to Viewer Integration > Defining the Start and End of a Trial
        Eyelink('Message', 'TRIAL_RESULT 0');
        WaitSecs(0.01); % Allow some time before ending the trial
        
        % Clear Screen() textures that were initialized for each trial iteration
        Screen('Close', backgroundTexture);        
        Screen('Close', imgTexture);
        Screen('Close', stimTexture);                
    end % End trial loop
    
    
    %% STEP 6: CLOSE EDF FILE. TRANSFER EDF COPY TO DISPLAY PC. CLOSE EYELINK CONNECTION. FINISH UP
    
    % Put tracker in idle/offline mode before closing file. Eyelink('SetOfflineMode') is recommended.
    % However if Eyelink('Command', 'set_idle_mode') is used, allow 50ms before closing the file as shown in the commented code:
    % Eyelink('Command', 'set_idle_mode');% Put tracker in idle/offline mode
    % WaitSecs(0.05); % Allow some time for transition  
    Eyelink('SetOfflineMode'); % Put tracker in idle/offline mode
    Eyelink('Command', 'clear_screen 0'); % Clear Host PC backdrop graphics at the end of the experiment
    WaitSecs(0.5); % Allow some time before closing and transferring file    
    Eyelink('CloseFile'); % Close EDF file on Host PC       
    % Transfer a copy of the EDF file to Display PC
    transferFile; % See transferFile function below  
catch % If syntax error is detected
    cleanup;
    % Print error message and line number in Matlab's Command Window
    psychrethrow(psychlasterror);
end
PsychPortAudio('Close', pahandle);
PsychPortAudio('Close', pamaster);


% Cleanup function used throughout the script above
    function cleanup
        sca; % PTB's wrapper for Screen('CloseAll') & related cleanup, e.g. ShowCursor
        Eyelink('Shutdown'); % Close EyeLink connection
        ListenChar(0); % Restore keyboard output to Matlab
        if ~IsOctave; commandwindow; end % Bring Command Window to front
    end

% Function for transferring copy of EDF file to the experiment folder on Display PC.
% Allows for optional destination path which is different from experiment folder
    function transferFile
        try
            if dummymode ==0 % If connected to EyeLink
                % Show 'Receiving data file...' text until file transfer is complete
                Screen('FillRect', window, el.backgroundcolour); % Prepare background on backbuffer
                Screen('DrawText', window, 'Receiving data file...', 5, height-35, 0); % Prepare text
                Screen('Flip', window); % Present text
                fprintf('Receiving data file ''%s.edf''\n', edfFile); % Print some text in Matlab's Command Window
                
                % Transfer EDF file to Host PC
                % [status =] Eyelink('ReceiveFile',['src'], ['dest'], ['dest_is_path'])
                status = Eyelink('ReceiveFile');
                
                % Check if EDF file has been transferred successfully and print file size in Matlab's Command Window
                if status > 0
                    fprintf('EDF file size: %.1f KB\n', status/1024); % Divide file size by 1024 to convert bytes to KB
                end
                % Print transferred EDF file path in Matlab's Command Window
                fprintf('Data file ''%s.edf'' can be found in ''%s''\n', edfFile, pwd);
            else
                fprintf('No EDF file saved in Dummy mode\n');
            end
            cleanup;
        catch % Catch a file-transfer error and print some text in Matlab's Command Window
            fprintf('Problem receiving data file ''%s''\n', edfFile);
            cleanup;
            psychrethrow(psychlasterror);
        end
    end
end
