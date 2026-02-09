function cc_regressModel_Photometry_LickingLama_AJ(fPath, optsIn)
% function to run linear encoding model on photometry data from puffy penguin paradigm
rng('default'); %for reproducability

if ~exist('optsIn', 'var')
    optsIn  = [];
end

%% define the local path
if fPath(end) ~= filesep
    fPath(end+1) = filesep;
end
savePath = fullfile(fPath, 'EncoderResults');
if ~exist(savePath, 'dir'); mkdir(savePath); end

%% define some variables
opts.skipIfExist = false; %flag to skip the current recording if encoder results already exist
opts.showOrthplot = false; %flag to show plot of regressor overlap
opts.preTrigDur = 0.5; %duration of baseline before trial onset
opts.postTrigDur = 8; %duration of trial after stimulus onset
opts.nrFolds = 20; %nr of folds for cross-validation
opts.testFrac = 0.1; %fraction of data used for testing in each fold
opts.removeAutoTrials = true; %flag to remove auto-rewarded trials. Maybe fine to keep them when focusing on sensory responses instead of choice.
opts.innateTask = false; %flag to use innate instead of audio task. The time of water presentation counts as stimulus
opts.videoDims = 100; %nr of video dimensions per type

% variables for stim regressors (use Type 2)
opts.preStimDur = 0; %duration of pre-stimulus period to include in stim regressors in seconds
opts.postStimDur = 1; %duration of post-stimulus period to include in non-initial stim regressors in seconds
opts.stimPeriodDur = 5.5; %duration of stimulus/delay period in seconds. Used for first stimulus in the sequence.

% variables for movement regressors (use Type 3)
opts.preMoveDur = 0.2; %duration of pre-movement period in seconds
opts.postMoveDur = 3; %duration of post-movement period in seconds

%variables to cross-validate
opts.cvRegs = {'time', 'choice', 'prevChoice', 'success', 'prevSuccess', 'water', 'licks', 'video'}; 

%check input opts and update opts fields if they were given
opts = copyCommonFields(optsIn, opts);

% skip model function if results exist already
if length(dir(savePath))>2 && opts.skipIfExist
    disp('GLM results already exist. Skipping recording');
    return
end

%% load photometry and behavioral data
[~, recName] = fileparts(fPath(1:end-1));
cRec = dir([fPath, '*', filesep, '*' recName '_2025*.mat']); %path to photometry data file
cPath = [cRec(1).folder filesep cRec(1).name];
cData = load(cPath); %load photometry and behavior
bhv = cData.bhv;
disp(['Photometry data: ', cPath]);

%% get frame specific information based on data sampling ate
frameRate = cData.sRate; %target frame rate. This should be 30Hz for photometry data.
opts.frameRate = frameRate;
opts.trialDur = opts.preTrigDur + opts.postTrigDur; %duration of a full trial
opts.framesPerTrial = opts.trialDur * opts.frameRate + 1; % nr. of frames per trial

opts.preTrig = opts.frameRate * opts.preTrigDur;
opts.postTrig = ceil(opts.frameRate * opts.postTrigDur);
opts.firstCuePreTime = ceil(opts.preStimDur * opts.frameRate);    %number of frames before first cue in the model
opts.firstCuePostTime = ceil(opts.stimPeriodDur * opts.frameRate); %number of frames after first cue
opts.otherCuePreTime = ceil(opts.preStimDur * opts.frameRate);  %number of frames before other cues 
opts.otherCuePostTime = ceil(opts.postStimDur * opts.frameRate); %number of frames after other cues

% variables for movement regressors (use Type 3)
opts.movePreTime = ceil(opts.preMoveDur * opts.frameRate);  % precede motor events to capture preparatory activity in frames (used for eventType 3)
opts.movePostTime = ceil(opts.postMoveDur * opts.frameRate);   % follow motor events for mPostStim in frames (used for eventType 3)

%% load video data and align photometry data to get consistent frame counts for both
fprintf('Load video data...');
vidPath = [fPath filesep];

cVid = dir([vidPath '*mergeCam*lowD.mat']);
load([vidPath cVid.name],'mergeV', 'vidTimes', 'vidTrigs');

cVid = dir([vidPath '*mergeMotionCam*lowD.mat']);
load([vidPath cVid.name], 'motionV');

frameTimes = vidTimes{1}; %frame times for video data
if frameRate ~= round(1/median(diff(frameTimes)))
    error('Video sampling rate does not match photometry data');
end

% check that tralcounts line up between trigers and behavioral data
nrTrials = length(bhv.Rewarded);
if length(vidTrigs{1}) ~= nrTrials
    % get trial numbers from video to match with behavior data
    cVid = dir([vidPath '*cam0*.avi']);
    [~, ~, ~, trialOn]  = mV_loadSyncPG(cVid.folder, 0, 2, true);
    bhv = selectBehaviorTrials(bhv, trialOn(:,3));
    nrTrials = length(bhv.Rewarded);
    
    useIdx = ismember(cData.trialNumbers, trialOn(:,3)');
    cData.trialOnTimes = cData.trialOnTimes(useIdx);
    cData.trialNumbers = cData.trialNumbers(useIdx);
    
    vidTrigs{1} = vidTrigs{1}(trialOn(:,3) <= nrTrials);
%     error('Number of trial triggers in video and behavioral data does not match.');
end

% get trial onset times
vidTrialOnFrames = nan(1, nrTrials);
for iTrials = 1 : nrTrials
    cTrig = find(vidTimes{1} > vidTrigs{1}(iTrials), 1); %this is the frame that was taken after which the trial started
    [~,b] = min(abs(vidTrigs{1}(iTrials) - vidTimes{1}(cTrig-1:cTrig))); %find closes trigger time
    vidTrialOnFrames(iTrials) = cTrig - 2 + b; 
end
fprintf(' done\n')

%% correct photometry timestamps to line up with video trigger times
[~, driftCorr] = makeCorrection(vidTrigs{1}', cData.trialOnTimes, false);
vcTime = applyCorrection(cData.blueFrameTimes, driftCorr); %adjust photometry times to align with video
vcTrigs = applyCorrection(cData.trialOnTimes, driftCorr); %adjust photometry times to align with video
Vc = cData.Vc;

% remove frames at the beginning
if vcTime(1) < frameTimes(1) %photometry was started first so some video data is missing. remove photometry data to align.
    onsetFrame = find(vcTime > frameTimes(1), 1); %find first photometry frame where video was running
    Vc = Vc(onsetFrame:end, :);
    vcTime = vcTime(onsetFrame:end, :);
    
else %same as above but video was running first
    onsetFrame = find(frameTimes > vcTime(1), 1); %find first photometry frame where video was running
    mergeV = mergeV(:, onsetFrame:end);
    motionV = motionV(:, onsetFrame:end);
    frameTimes = frameTimes(onsetFrame:end);
    vidTrialOnFrames = vidTrialOnFrames - onsetFrame + 1;
%     error('!!! Video was started before the photometry. This as not been tested yet. Check that this works properly !!!');
end

% remove frames at the end
if max(frameTimes) < max(vcTime) %behavior was stopped before photometry. Remove some photometry frames at the end.
    lastFrame = find(vcTime > max(frameTimes), 1) - 1; %find first photometry frame before video was not running anymore
    Vc = Vc(1:lastFrame, :);
    vcTime = vcTime(1:lastFrame);   
    
else %same as above but video was stopped first
    lastFrame = find(frameTimes > max(vcTime), 1) - 1; %find first photometry frame before video was not running anymore
    mergeV = mergeV(:, 1:lastFrame);
    motionV = motionV(:, 1:lastFrame);
    frameTimes = frameTimes(1:lastFrame);   
end

% add one frame at the end of photometry to have same time range for photometry and video. This is needed for the resampling.
Vc = cat(1, Vc(1,:), Vc, Vc(end,:));
vcTime = cat(1, frameTimes(1), vcTime, frameTimes(end)+1E-3);
Vc = mV_vidResamp(Vc, vcTime', frameRate); %resample to get uniform sampling that matches the video
vcTime = (0 : size(Vc,1)-1) ./ frameRate; %new time to with resampled Vc data
   
% %% remove trials with missing frames and auto trials from the data
% useIdx = ~any(isnan(squeeze(motionV(1,:,:))), 1) & ~any(isnan(squeeze(Vc(1,:,:))), 1);
% 
% % remove auto-reward trials
% if opts.removeAutoTrials
%     useIdx = useIdx & bhv.Assisted;
% end
% 
% stimTimes = stimTimes(useIdx);
% mergeV = mergeV(:,:,useIdx); %only keep self-performed trials
% motionV = motionV(:,:,useIdx); %only keep self-performed trials
% Vc = Vc(:,:,useIdx); %only keep self-performed trials
% bhv = selectBehaviorTrials(bhv, useIdx);

%% equalize L/R choices
useIdx = true(1, length(bhv.Rewarded)); %use all trials for now
useIdx(1:10) = false; %dont use the first 10 trials because they are sometimes affected by fluorescence drift
% choiceIdx = cc_equalizeTrials(useIdx, bhv.ResponseSide == 1, bhv.Rewarded, inf, true); %equalize correct L/R choices
bhv = selectBehaviorTrials(bhv, useIdx);
nrTrials = sum(useIdx);

%% check behavioral data for some markers of interest
trialTimes = vidTrialOnFrames(useIdx); %time of trial onset

water = NaN(1, nrTrials);
firstLickR = cell(1, nrTrials);
nextLickR = cell(1, nrTrials);
firstStimIdx = cell(1,2);
stimTimes = nan(1,nrTrials);
punishTime = nan(1,nrTrials);
for iTrials = 1 : nrTrials
    
    stimTimes(iTrials) = bhv.RawEvents.Trial{iTrials}.States.PlayStimulus(1);
    stimTimes(iTrials) = floor(stimTimes(iTrials) * frameRate) + trialTimes(iTrials);
        
    % check for licks
    if isfield(bhv.RawEvents.Trial{iTrials}.Events,'TouchShaker1_2') %check for right licks
        cData = bhv.RawEvents.Trial{iTrials}.Events.TouchShaker1_2;
        cData = unique(floor(cData * frameRate) + trialTimes(iTrials)); %change to absolute frame times
        if ~isempty(cData)
            firstLickR{iTrials} = cData(1); cData(1) = [];
            nextLickR{iTrials} = cData;
        end
    end
    
    % check outcomes
    if ~isnan(bhv.RawEvents.Trial{iTrials}.States.Reward(1)) %check for reward state
        water(iTrials) = bhv.RawEvents.Trial{iTrials}.States.Reward(1);
        water(iTrials) = floor(water(iTrials) * frameRate) + trialTimes(iTrials);
    end
    
    if ~isnan(bhv.RawEvents.Trial{iTrials}.States.HardPunish(1)) %check for punish state
        punishTime(iTrials) = bhv.RawEvents.Trial{iTrials}.States.HardPunish(1);
        punishTime(iTrials) = floor(punishTime(iTrials) * frameRate) + trialTimes(iTrials);
    end
end

%% make design matrices - cognitive regressors
cogLabels = {'choice', 'prevChoice', 'success', 'prevSuccess'};
% cogLabels = {'choiceL', 'choiceR'};
events = false(size(Vc,1), length(cogLabels));
eventType = ones(1,length(cogLabels));

% events(:,1) = makeLogical(trialTimes, size(Vc,1)); %time
events(:,1) = makeLogical(trialTimes(bhv.DidNotChoose == 0), size(Vc,1)); %GO trials

trialIdx = find(bhv.DidNotChoose == 0)+1; %trials where previous choice was GO
trialIdx = trialIdx(trialIdx <= nrTrials);
% events(:,2) = makeLogical(trialTimes(trialIdx), size(Vc,1)); % previous trial was GO

events(:,3) = makeLogical(trialTimes(bhv.Rewarded), size(Vc,1)); %rewarded

trialIdx = find(bhv.Rewarded)+1; %trials where previous choice was rewarded
trialIdx = trialIdx(trialIdx <= nrTrials);
% events(:,4) = makeLogical(trialTimes(trialIdx), size(Vc,1)); % previous trial rewarded

[cogR, cogIdx] = makeDesignMatrix_noTrials(events, eventType, cogLabels, opts);

%% make design matrices - stimulus regressors
stimFreqs = unique(bhv.audioFreq);
stimLabels = compose('audioStim_%dkHz', stimFreqs);
stimLabels = [stimLabels, {'water', 'punish'}];
eventType = repmat(3,1, length(stimLabels));
events = false(size(Vc,1), length(stimLabels));

% find stimulus events
for iFreqs = 1 : length(stimFreqs)
    cIdx = bhv.audioFreq == stimFreqs(iFreqs);
    events(:,iFreqs) = makeLogical(stimTimes(cIdx), size(Vc,1)); %frequencys
end

%find outcome events
events(:,  length(stimFreqs)+1) = makeLogical(water, size(Vc,1)); %water events
events(:,  length(stimFreqs)+2) = makeLogical(punishTime, size(Vc,1)); %punish events

opts.mPreTime = opts.otherCuePreTime;
opts.mPostTime = opts.otherCuePostTime; 
[stimR, stimIdx] = makeDesignMatrix_noTrials(events, eventType, stimLabels, opts);
stimIdx = stimIdx  + max(cogIdx);

%% add dummy variable for shuffle control - based on timeR but shuffled for each regressor
dummyLabel = {'dummy'};
events = makeLogical(trialTimes, size(Vc,1)); %time
[dummyR, dummyIdx] = makeDesignMatrix_noTrials(events, 3, dummyLabel, opts);
dummyIdx = dummyIdx  + max(stimIdx);

%shuffle dummy regressors
for iCol = 1:size(dummyR,2)
    dummyR(:,iCol) = dummyR(randperm(size(dummyR,1)),iCol);
end

%% make design matrices - movement regressors
moveLabels = {'firstLicksR', 'nextLicksR'};
eventType = repmat(3,1,length(moveLabels));
events = false(size(Vc,1), length(moveLabels));

events(:,1) = makeLogical(cat(2,firstLickR{:}), size(Vc,1)); % right licks
events(:,2) = makeLogical(cat(2,nextLickR{:}), size(Vc,1)); % next right licks

opts.mPreTime = opts.movePreTime; 
opts.mPostTime = opts.movePostTime;
[moveR, moveIdx] = makeDesignMatrix_noTrials(events, eventType, moveLabels, opts);
moveIdx = moveIdx  + max(dummyIdx);

% video regressors
vidLabels = {'video' ,'motionVideo'};
vidR = [reshape(mergeV(1:opts.videoDims,:), opts.videoDims, [])', reshape(motionV(1:opts.videoDims,:), opts.videoDims, [])'];
vidIdx = [ones(opts.videoDims, 1); ones(opts.videoDims, 1)+1] + max(moveIdx);   

%% merge all into one design matrix fullR
fullR = [cogR, stimR, dummyR, moveR, vidR];
regIdx = [cogIdx; stimIdx; dummyIdx; moveIdx; vidIdx];
regLabels = [cogLabels, stimLabels, dummyLabel, moveLabels, vidLabels];

% run QR and check for rank-defficiency
rejIdx = nansum(abs(fullR)) < 10;
[~, fullQRR] = qr(bsxfun(@rdivide,fullR(:,~rejIdx),sqrt(sum(fullR(:,~rejIdx).^2))),0); %orthogonalize design matrix

% check if regressors need to be removed if design matrix is rank-defficient
showOrthplot = false;
if sum(abs(diag(fullQRR)) > max(size(fullR)) * eps(fullQRR(1))) < size(fullQRR,2) %check if design matrix is full rank
    temp = ~(abs(diag(fullQRR)) > max(size(fullR)) * eps(fullQRR(1)));
    fprintf('Design matrix is rank-defficient. Removing %d/%d additional regressors.\n', sum(temp), sum(~rejIdx));
    rejIdx(~rejIdx) = temp; %reject regressors that cause rank-defficint matrix
    showOrthplot = true;
end

% show design matrix angles
if opts.showOrthplot || showOrthplot
    figure; plot(abs(diag(fullQRR))); ylim([0 1.1]); title('Regressor orthogonality'); drawnow; %this shows how orthogonal individual regressors are to the rest of the matrix
end

% reject regressors that are too sparse or rank-defficient
fullR(:,rejIdx) = []; %clear empty regressors
fprintf(1, 'Rejected %d/%d empty or redundant regressors\n', sum(rejIdx),length(rejIdx));

%% orthogonalize video against other movements
[Q, ~] = qr([moveR, vidR], 0);
fullR(:, end-size(vidR, 2) : end) = Q(:, end-size(vidR, 2) : end);
save([savePath filesep 'rawVidR.mat'], 'vidR', 'vidIdx', 'vidLabels');

%% run ridge regression
% make sure neural data is in the right format, detrend and smooth
startIdx = find(sum([cogR, stimR] == 1, 2), 1); %start with first frame where a task regressor is found
Vc(1:startIdx, :) = 0; %dont use frames before task regressors exist
fullR(1:startIdx, :) = 0; %dont use frames before task regressors exist
Vc = smoothCol(Vc, 1, 5, 'box');
newVc = bsxfun(@minus, Vc, mean(Vc,1)); %should be zero-mean

disp('Fitting ridge model');
[ridgeVals, dimBeta] = gT_ridgeMML(newVc, fullR, true); %get ridge penalties and beta weights.
fprintf('Mean ridge penalty for original video, zero-mean model: %f\n', mean(ridgeVals));
save([savePath filesep 'dimBeta.mat'], 'dimBeta', 'ridgeVals');
save([savePath filesep 'regData.mat'], 'fullR', 'rejIdx' ,'regIdx', 'regLabels', 'fullQRR', 'bhv', 'trialTimes', 'stimTimes', 'opts', '-v7.3');
save([savePath filesep 'rawVc.mat'], 'newVc');

%% cross-validation
%reduce data size to trials so that movements dont get too much importance vs task
trialIdx = (trialTimes' + (0:opts.framesPerTrial-1))';
trialIdx = trialIdx(trialIdx<=size(Vc,1));

fprintf('Cross-validating full model...');
[fullCorr, ~, fullBeta] =  cc_overlapCrossValModel_SM(fullR(trialIdx(:),:), newVc(trialIdx(:),:), regLabels, regIdx(~rejIdx), regLabels, opts.nrFolds, opts.testFrac, opts.framesPerTrial);
save([savePath filesep 'fullCorr.mat'], 'fullCorr', 'fullBeta', 'trialIdx');
fprintf('done\n');

%% check for movement and task explained variance    
fprintf('Cross-validating movement/task models...');

% motor
motorLabels = [moveLabels, vidLabels];
[motorCorr, motorR, motorBeta] =  cc_overlapCrossValModel_SM(fullR(trialIdx(:),:), newVc(trialIdx(:),:), motorLabels, regIdx(~rejIdx), regLabels, opts.nrFolds, opts.testFrac, opts.framesPerTrial);
save([savePath filesep 'motorCorr.mat'], 'motorCorr', 'motorBeta', 'motorLabels', 'rejIdx' ,'regIdx', 'regLabels');

% task
taskLabels = [cogLabels, stimLabels];
[taskCorr, taskR, taskBeta] =  cc_overlapCrossValModel_SM(fullR(trialIdx(:),:), newVc(trialIdx(:),:), taskLabels, regIdx(~rejIdx), regLabels, opts.nrFolds, opts.testFrac, opts.framesPerTrial);
save([savePath filesep 'taskCorr.mat'], 'taskCorr', 'taskBeta', 'taskLabels', 'rejIdx' ,'regIdx', 'regLabels');

% dummy control
nonDummyLabels = regLabels(~contains(regLabels, 'dummy'));
nonDummyCorr =  cc_overlapCrossValModel_SM(fullR(trialIdx(:),:), newVc(trialIdx(:),:), nonDummyLabels, regIdx(~rejIdx), regLabels, opts.nrFolds, opts.testFrac, opts.framesPerTrial);
save([savePath filesep 'nonDummyCorr.mat'], 'nonDummyCorr');
fprintf('done\n');

%% check all regressors individually - this takes a while
% fprintf('Cross-validating other variables...');
% for iRegs = 1 : length(opts.cvRegs)
%     
%     nonCregLabels = regLabels(~contains(regLabels, opts.cvRegs{iRegs}, 'IgnoreCase', true));
%     cVar = ['non' opts.cvRegs{iRegs} 'Corr'];
%     eval([cVar ' =  cc_overlapCrossValModel_SM(fullR(trialIdx(:),:), newVc(trialIdx(:),:), nonCregLabels, regIdx(~rejIdx), regLabels, opts.nrFolds, opts.testFrac, opts.framesPerTrial);']);
%     save([savePath filesep cVar '.mat'], cVar);
%     
% end
% fprintf('done\n');

