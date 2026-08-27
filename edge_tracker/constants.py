# Camera credentials belong in backend/.env.  Keeping an empty CLI fallback
# prevents them from appearing in source control, process listings, or logs.
DEFAULT_RTSP_URL = ""
DISPLAY_SIZE = (1280, 720)
TACTICAL_MAP_SIZE = 600
DEFAULT_TACTICAL_MAP_SIZE_CM = 480
DEFAULT_TACTICAL_MAP_GRID_COLUMNS = 5
DEFAULT_TACTICAL_MAP_GRID_ROWS = 5
LEFT_ANKLE_KEYPOINT_INDEX = 15
RIGHT_ANKLE_KEYPOINT_INDEX = 16
MIN_ANKLE_CONFIDENCE = 0.35
DEFAULT_FUSION_DISTANCE_CM = 50.0
DEFAULT_MEDIAPIPE_MODEL_PATH = "pose_landmarker_heavy.task"
MEDIAPIPE_LEFT_ANKLE = 27
MEDIAPIPE_RIGHT_ANKLE = 28
MEDIAPIPE_LEFT_HEEL = 29
MEDIAPIPE_RIGHT_HEEL = 30
MEDIAPIPE_LEFT_FOOT_INDEX = 31
MEDIAPIPE_RIGHT_FOOT_INDEX = 32
MIN_MEDIAPIPE_VISIBILITY = 0.45
MEDIAPIPE_NOSE = 0
MEDIAPIPE_LEFT_EYE = 2
MEDIAPIPE_RIGHT_EYE = 5
MEDIAPIPE_LEFT_SHOULDER = 11
MEDIAPIPE_RIGHT_SHOULDER = 12
MEDIAPIPE_LEFT_HIP = 23
MEDIAPIPE_RIGHT_HIP = 24
MEDIAPIPE_LEFT_KNEE = 25
MEDIAPIPE_RIGHT_KNEE = 26
MEDIAPIPE_LEFT_EAR = 7
MEDIAPIPE_RIGHT_EAR = 8
MEDIAPIPE_MOUTH_LEFT = 9
MEDIAPIPE_MOUTH_RIGHT = 10
MIN_ANATOMICAL_ANCHOR_PIXELS = 4.0
MIN_ANATOMICAL_FULL_BODY_PIXELS = 25.0
MIN_ANATOMICAL_RATIO = 0.04
MAX_ANATOMICAL_RATIO = 0.17
MIN_INITIAL_FOOT_VISIBILITY = 0.70
HEAD_DOWN_ANCHOR_FRACTION = 0.35
MIN_HEAD_DOWN_PIXELS = 6.0
DEFAULT_ANATOMICAL_RATIO_EMA_ALPHA = 0.02
DEFAULT_MAX_FOOT_JUMP_PIXELS_PER_FRAME = 120.0
DEFAULT_MAX_PERSON_SPEED_MPS = 8.0
DEFAULT_MAP_POSITION_EMA_ALPHA = 0.35
DEFAULT_POSE_DROPOUT_TTL_FRAMES = 30

# --- Position quality -------------------------------------------------------
# A ground point is only as good as the way it was obtained, and two consumers
# want opposite things from it.  The tactical map wants the *most accurate*
# point and may freely take it from whichever camera sees best.  Cross-camera
# association wants an *independent* point per camera, because a point borrowed
# from the other camera would make the matcher's own output its input.  These
# grades let both consumers ask the same question and act on it differently.
#
# HARD  -- feet actually measured, in an unobstructed box.  May veto.
# SOFT  -- inferred from the learned body ratio, or measured in a box whose
#          bottom is clipped or occluded.  May sustain a pairing, never break
#          one, because the point may simply not be where the person stands.
# STALE -- coasted from an earlier frame; no measurement at all this frame.
POSITION_QUALITY_HARD = "hard"
POSITION_QUALITY_SOFT = "soft"
POSITION_QUALITY_STALE = "stale"
POSITION_QUALITY_NONE = "none"

# Relative say each grade gets in the fused map point.  Deliberately continuous
# rather than a three-way branch: a camera losing sight of the feet decays
# toward zero influence instead of the dot stepping across a threshold, which
# reads to an operator as real movement.  The hard:soft ratio of 10:1 means a
# blind camera disagreeing by a metre still moves the dot under 10 cm.
POSITION_QUALITY_FUSION_WEIGHTS = {
    POSITION_QUALITY_HARD: 1.0,
    POSITION_QUALITY_SOFT: 0.10,
    POSITION_QUALITY_STALE: 0.02,
    POSITION_QUALITY_NONE: 0.0,
}

# How much of the usual smoothing step a point of each grade is allowed to
# take.  Separate from the fusion weights above because the questions differ:
# fusion asks "whose point do we believe *this frame*", where a near-total
# preference is right, while the filter asks "how much should this point move
# a position we will keep using", where a near-total refusal would leave a
# genuinely-moving person's dot stuck behind them whenever their feet are hard
# to see.  Stale points contribute nothing -- they are a previous position
# being handed back, so folding one in again only double-counts it.
POSITION_QUALITY_SMOOTHING_SCALE = {
    POSITION_QUALITY_HARD: 1.0,
    POSITION_QUALITY_SOFT: 0.35,
    POSITION_QUALITY_STALE: 0.0,
    POSITION_QUALITY_NONE: 0.0,
}

# How far a soft point's association gate is widened.  The body-ratio estimate
# extrapolates foot position from the nose-to-shoulder span, so it multiplies
# pixel noise by 1/ratio -- up to 25x at the configured floor.  Holding it to
# the same tolerance as a measured foot fix is what drops people out of the
# cross-camera match the moment their feet are hidden.
SOFT_POSITION_GATE_MULTIPLIER = 2.5

# Consecutive frames of disagreement required before a fused pair is broken or
# an identity is split.  Anything shorter and ordinary homography distortion at
# a grazing angle -- which produces points graded HARD, because the camera has
# no way to know it is wrong -- flickers the dot and the headcount every frame.
# Asymmetric on purpose: a wrong split puts a duplicate person on the map
# immediately, whereas a wrong merge is still caught by appearance later.
DEFAULT_POSITION_SPLIT_FRAMES = 6
# A track trying to *join* a master used to get no patience at all: one reading
# over the limit vetoed the match outright.  That is the wrong way round.  A
# binding already working is protected for six frames because breaking it puts
# a duplicate on the map -- and refusing to form it in the first place puts the
# same duplicate there, permanently, with no audit path back.  One recorded
# session denied a 0.2175 appearance match (threshold 0.30) on a single reading
# 62.4cm apart against a 50cm limit, 28ms of skew, and minted a second ID for a
# man who already had one.  A wrong join is still revoked six frames later by
# the established-binding rule above; a wrong new master is forever.
#
# The patience is deliberately spent only on *marginal* disagreements, because
# the magnitude is what separates the two failures.  Being a quarter over the
# limit is the homography's error bar.  Being several times over is a different
# person, and forgiving that would let one master be held by two tracks metres
# apart.  Measured: the disagreement that cost a real ID was 1.25x the limit,
# while every case that must still be refused on sight runs 2.4x to 6x.
DEFAULT_NEW_MATCH_POSITION_SPLIT_FRAMES = 2
DEFAULT_NEW_MATCH_POSITION_SLACK_RATIO = 1.5
DEFAULT_POSITION_PAIR_FRAMES = 2

# Ordering of the grades, so callers can ask "which of these two points do we
# believe more" without hard-coding the vocabulary.
POSITION_QUALITY_RANK = {
    POSITION_QUALITY_NONE: 0,
    POSITION_QUALITY_STALE: 1,
    POSITION_QUALITY_SOFT: 2,
    POSITION_QUALITY_HARD: 3,
}

# Two foot centroids closer than this are not two standing people.  Shoulder
# width alone is around 45 cm, so a pair of adults cannot place the centres of
# their feet 35 cm apart.  Used only to decide whether the *display* is drawing
# one person twice; it never merges identities and never reaches ReID.
#
# Deliberately far tighter than DEFAULT_FUSION_DISTANCE_CM.  That limit answers
# "could these be the same person?", which tolerates real measurement error.
# This one answers "is it physically impossible for these to be two people?",
# which must not.
DEFAULT_DISPLAY_DUPLICATE_DISTANCE_CM = 35.0
DEFAULT_YOLO_NMS_IOU = 0.55
# OC-SORT associates using travel direction as well as box overlap, which is
# what BoT-SORT lacked when two people crossed and their IDs traded.  The
# BoT-SORT file is kept alongside it; pass --tracker-config
# bytetrack_ghost_resistant.yaml to compare the two on the same footage.
DEFAULT_TRACKER_CONFIG_PATH = "ocsort_ghost_resistant.yaml"
DEFAULT_REID_DISTANCE_THRESHOLD = 0.30
DEFAULT_REID_SIMILARITY_THRESHOLD = 1.0 - DEFAULT_REID_DISTANCE_THRESHOLD
DEFAULT_REID_MEMORY_TTL_FRAMES = 450
# The TTL above answers "this track came back after a gap -- still the same
# person?", which deserves to be generous.  This answers a different question:
# "is this track ever coming back at all?"  The tracker's own ``track_buffer``
# is 30 frames, after which it retires the number and issues a new one, so a
# binding held past that is held for an ID that can never return.  A track
# absent this long is struck off the group it belonged to; without that, a
# member which vanished mid-intake blocks its group's promotion forever and
# everyone in it stays "analysing".  Kept above track_buffer so the identity
# layer never gives up on a track the tracker still considers alive.
DEFAULT_TRACK_ABANDON_FRAMES = 45
DEFAULT_REID_EMA_ALPHA = 0.2
# A newly spawned local track is treated as a possible ByteTrack shadow only
# when it almost completely overlaps an older track in the same camera.
# Appearance still has the final say before an ID handoff is committed.
DEFAULT_REID_SHADOW_IOU_THRESHOLD = 0.65
DEFAULT_REID_SHADOW_CONTAINMENT_THRESHOLD = 0.85
DEFAULT_REID_SHADOW_CENTER_DISTANCE_RATIO = 0.20
DEFAULT_REID_SHADOW_PROBATION_FRAMES = 3
DEFAULT_REID_SHADOW_SEPARATION_FRAMES = 2
REID_GALLERY_SLOTS = ("baseline", "front", "back", "left_side", "right_side")
REID_SEMANTIC_SLOTS = REID_GALLERY_SLOTS[1:]
# Measured against the 27 crops one recorded session filed into galleries: the
# three a person would object to on sight scored 102, 115 and 119, the next two
# 174 and 176, and the rest ran from 212 to 1755 with a median of 739.  100 let
# a crop through whose face was a smear of pixels, and it then sat in a gallery
# arguing about who its owner was.  200 clears that group and keeps 22 of the
# 27.  Higher starts refusing merely mediocre crops, and a starved gallery has
# its own failure -- people stranded in "analysing" because nothing arrives.
# The intake timeout still relaxes this gate, so a person who never produces a
# sharp crop is delayed rather than stranded.
DEFAULT_REID_BLUR_THRESHOLD = 200.0
DEFAULT_REID_SEMANTIC_CONFIDENCE = 0.75
DEFAULT_REID_INTAKE_DELAY_SECONDS = 0.5
DEFAULT_REID_INTAKE_TIMEOUT_SECONDS = 5.0
DEFAULT_REID_SEMANTIC_COOLDOWN_FRAMES = 30
DEFAULT_REID_SEMANTIC_RETRY_FRAMES = 10
DEFAULT_REID_QUEUE_SIZE = 64
DEFAULT_REID_INTAKE_RETRY_FRAMES = 15
DEFAULT_REID_MAX_RETRY_FRAMES = 240
DEFAULT_REID_CROP_SIDE_PADDING = 0.02
DEFAULT_REID_CROP_TOP_PADDING = 0.02
DEFAULT_REID_CROP_BOTTOM_PADDING = 0.10
DEFAULT_REID_CROP_MAX_INTRUDER_AREA_RATIO = 0.15
# The budget above assumes an intruder's box is full of the intruder.  That
# holds for someone standing in front, who really does replace the body the
# crop exists to show.  It is wrong for someone standing behind: the subject
# hides most of that box, so what the gate measures is largely the subject's
# own pixels, and charging it at the front rate stranded people in
# "analysing".  One recorded run logged 14,148 overlap rejections against
# 3,931 accepted crops -- 22x every other intake rejection combined.
# Deliberately a looser budget rather than an exemption: a person behind and
# offset to one side still shows an arm or a leg beside the subject, and those
# pixels do reach the crop.  Starting at twice the front budget, to be retuned
# once a session's logs show the behind-intruder ratio distribution.
DEFAULT_REID_CROP_MAX_BEHIND_INTRUDER_AREA_RATIO = 0.30
# How much higher an intruder's feet must sit before it counts as clearly
# behind, as a fraction of the subject's box height.  Box bottoms jitter by a
# few pixels between frames and a bare comparison would flip a borderline pair
# between the two budgets frame to frame.  Depth read this way only holds
# while everyone stands on one floor, which is the tent's normal case.
DEFAULT_REID_CROP_BEHIND_FOOT_MARGIN = 0.05
DEFAULT_REID_FRAME_EDGE_MARGIN_PIXELS = 2.0
DEFAULT_REID_ROLE_CHECKPOINT = "evacuation_mobilenet_v1.pth"
DEFAULT_REID_ROLE_CONFIDENCE = 0.53
DEFAULT_CROSS_CAMERA_MAX_SKEW_SECONDS = 0.35
# A location pair becomes one provisional person after a short, repeated
# agreement.  ReID can confirm it sooner, while a longer clean location
# streak provides a fallback when the cameras only see opposite body angles.
DEFAULT_PROVISIONAL_PAIR_FRAMES = 3
DEFAULT_PROVISIONAL_LOCATION_CONFIRM_FRAMES = 12
DEFAULT_PROVISIONAL_HOLD_GRACE_FRAMES = 5
DEFAULT_PROVISIONAL_HOLD_MAX_FRAMES = 12
# A physical-position wobble can split one member out of a temporary
# cross-camera group just before that member allocates the permanent master.
# Keep the remaining singleton recoverable for a short, bounded window so the
# original pair can be re-established and checked against that new master.
# After the window it is released into ordinary intake instead of displaying
# "Analyzing" for the rest of the track's lifetime.
DEFAULT_PROVISIONAL_SPLIT_RECOVERY_SECONDS = 10.0
DEFAULT_PROVISIONAL_CHALLENGE_DISTANCE = 0.55
# The stable-location shortcut promotes a pair without ever comparing the two
# cameras' crops to each other, so two people walking together while mutually
# occluded can be fused into one master.  Before that promotion the per-camera
# baselines are compared across viewing angles and the promotion is withheld
# when they disagree this strongly.  The gate only vetoes -- it never confirms
# -- because opposing cameras routinely see one person from opposite angles,
# so agreement cannot be required, only clear disagreement acted upon.
DEFAULT_PROVISIONAL_BASELINE_VETO_DISTANCE = 0.55
# After two people cross and swap local tracks, the tracker can capture a
# clean, sharp, fully visible crop of the WRONG person and file it under the
# original master.  No quality gate sees anything wrong with that crop, so a
# new angle is also compared against the views already stored for the identity
# and refused when it agrees with none of them.  Slot quality alone decides
# which of two admitted crops wins; this decides whether a crop may compete.
#
# Measured, not reasoned: across 34 admissions in one recorded session the
# genuine crops scored 0.110-0.351 against the gallery they were joining,
# while the three that put a second man into someone else's gallery scored
# 0.436, 0.451 and 0.475.  0.40 sits in the gap and refuses exactly those
# three.  An earlier 0.55 was a guess made before any of this was measured and
# admitted all of them.  Note the scale belongs to the median comparison used
# in _gallery_admission_rejected_locked -- against the closest stored view
# alone these same crops score far lower, so the two are not interchangeable.
DEFAULT_GALLERY_ADMISSION_DISTANCE = 0.40
# Folding a whole provisional group into an existing master used to turn on the
# single closest slot pair being under DEFAULT_REID_DISTANCE_THRESHOLD.  One
# man merged into another's identity on 0.28993 against a 0.30 bar -- and it
# was not a fluke, the borderline guard made him win three separate intake
# batches first, all at 0.26-0.29.  Across the target's whole gallery the same
# pair sat at 0.378, while every genuinely different pair in that session ran
# 0.53-0.79 and the one real duplicate ran 0.279.  The evidence to refuse it
# was there; a single best-slot comparison discarded it.
#
# 0.35 rather than 0.30 because one clean identity's own photographs spanned a
# median of 0.343 -- front and back of one person really are that far apart --
# and a threshold under that would refuse to reunite somebody with a wide
# viewpoint spread, which produces the duplicate IDs this is meant to prevent.
# Provisional: two labelled merges is not a calibration.
DEFAULT_PROVISIONAL_MERGE_DISTANCE = 0.35
# Local trackers swap boxes when two people cross, which silently transplants
# both master IDs.  Bound tracks are therefore re-checked against every gallery
# on this interval so the map repairs itself instead of drifting.
DEFAULT_IDENTITY_AUDIT_INTERVAL_SECONDS = 5.0
# Repair must be slower to act than it is to notice.  A contradicting master
# has to win by this margin, and win on this many consecutive audits, before a
# binding moves -- otherwise two people in similar clothing trade IDs forever.
DEFAULT_IDENTITY_AUDIT_MARGIN = 0.10
DEFAULT_IDENTITY_AUDIT_ROUNDS = 2
# A contest compares two claimants head to head with several crops each; the
# audit judges one track on one crop.  When both are working on the same track
# the audit stands aside, or it reassigns a claimant mid-arbitration and the
# contest is left waiting on a participant that has wandered off -- which cost
# 35 seconds of wrong IDs in one recorded huddle.  Standing aside forever is
# the worse failure though: a contest starved of clean crops would mute the
# only other repair path indefinitely, so patience is bounded.
DEFAULT_IDENTITY_AUDIT_CONTEST_PATIENCE_SECONDS = 30.0
# A confirmed master claimed at incompatible cross-camera locations is held
# until both claimants provide several clean, overlap-approved ReID crops.
DEFAULT_PHYSICAL_CONFLICT_REID_FRAMES = 3
# How much closer the winner must be than the loser before an arbitration is
# allowed to take a master off somebody.  0.05 was small enough that "both of
# these look like him" was read as a verdict.  Measured across two recorded
# sessions: the four verdicts that were right separated by 0.341, 0.406, 0.509
# and 0.516, while the two that evicted the real owner and produced a duplicate
# separated by 0.068 and 0.104.  Nothing observed lands between 0.104 and 0.341,
# so 0.15 sits in that gap with headroom on both sides.  Falling short is not a
# refusal -- the contest is recorded inconclusive and retried on fresh crops,
# which is the correct reading of two candidates the model cannot separate.
DEFAULT_PHYSICAL_CONFLICT_REID_MARGIN = 0.15
# Arbitration is not gallery admission.  The blur gate exists to keep a smeared
# face out of permanent storage; an arbiter only asks which of two candidates
# is nearer, and the winner still has to clear DEFAULT_PHYSICAL_CONFLICT_REID_
# MARGIN, so a mushy feature reads as inconclusive rather than as a wrong
# revocation.  Holding both to 200 deadlocked a real contest: one camera sat at
# 3/3 crops while the other was refused at 55-100 sharpness for 4.1 seconds,
# and the master it locked was denied to its true owner -- who was matching it
# at 0.047 -- for that whole time.  Intake already relaxes this gate after its
# timeout; arbitration needs the same release much sooner, because a contest is
# meant to be settled within a few frames rather than over a person's whole
# intake.  Below ~1s a slow camera would relax before a sharp one had a fair
# chance to answer cleanly.
DEFAULT_PHYSICAL_CONFLICT_BLUR_TIMEOUT_SECONDS = 1.5
# When even the relaxed gate produces nothing -- a candidate the detector keeps
# clipping, or one that has stopped yielding crops at all -- the hold can never
# conclude, and a location hold with no verdict blocks the appearance contest
# that would have returned the master to its owner.  Past this age a starved
# hold stands aside for a strong challenger.  It must exceed the blur timeout
# above, so relaxation gets its chance before the hold is written off.
DEFAULT_PHYSICAL_CONFLICT_STALL_SECONDS = 2.5
# A track rejected from one master waits briefly for another same-camera
# physical conflict to finish before it is allowed to create a brand-new ID.
# Once linked, the hold may survive a longer clean-crop wait, but is still
# bounded so a genuinely new person cannot remain unnumbered forever.
DEFAULT_PHYSICAL_CONFLICT_RECOVERY_GRACE_FRAMES = 15
DEFAULT_PHYSICAL_CONFLICT_RECOVERY_MAX_FRAMES = 450
FPS_EMA_ALPHA = 0.1
# EXPERIMENTAL two-plane metrology ("3D Level Detection"). Lives here so the
# tracker can expose the tunable without importing the experimental module on
# runs where the feature is switched off.
DEFAULT_THREE_D_MAX_LEAN_DEGREES = 6.0
DEFAULT_ELEVATED_MATRIX_1 = "homography_matrix_elevated.json"
DEFAULT_ELEVATED_MATRIX_2 = "homography_matrix_2_elevated.json"

# --- MiVOLO age/gender estimation -------------------------------------------
# MiVOLO reads a face crop and a body crop as two complementary inputs.  Its
# own pipeline feeds it a tight detection box with the face cut out of the body
# branch, so these constants exist to reproduce that framing from the crops the
# ReID intake already collects rather than to invent a new one.
#
# A face is worth far more than a body to an age estimate, so the crops handed
# to the model are ranked by how much face they actually contain, not by the
# order they arrived in.  Five is kept from the intake burst size; the model is
# run as one batch, which measured 67ms against 96ms for the same five crops
# one at a time.
DEFAULT_DEMOGRAPHICS_CROP_COUNT = 5
# Upstream MiVOLO refuses person crops below 50 px on either side.  Anything
# smaller carries no usable clothing or build detail once letterboxed to 384.
DEFAULT_DEMOGRAPHICS_MIN_BODY_PIXELS = 50
# The same argument applied to the face branch, at the scale a face occupies.
# Below roughly 24 px the resize to 384 is inventing detail, and a face that
# small predicts worse than passing no face at all -- which is a supported
# input, because the model was trained with a zeroed face tensor for exactly
# this case.
DEFAULT_DEMOGRAPHICS_MIN_FACE_PIXELS = 24
# How much a face-less reading counts against one with a face.  MiVOLO's own
# published figures put body-only age error several years above face-plus-body,
# so a body-only vote is kept as evidence but must not outweigh a clear look at
# the person's face.
DEFAULT_DEMOGRAPHICS_BODY_ONLY_WEIGHT = 0.35
# Winsorizing distance for the age vote, in years.  Taken from MiVOLO's own
# aggregate_votes_winsorized default: readings further than this from the
# median are pulled back to it rather than dropped, so one bad crop shifts the
# answer slightly instead of dragging the mean.
DEFAULT_DEMOGRAPHICS_MAX_AGE_SPREAD_YEARS = 6.0
# Demographics used to run once, on whatever the person looked like the moment
# they were first tracked -- which for someone who entered at the far end of the
# room is five forty-pixel faces, kept for the rest of the session.  A later,
# much closer look now re-runs the estimate.  The bar is deliberately a large
# multiple rather than any improvement: re-running on a marginally better crop
# would spend GPU on answers that cannot meaningfully change, and would let the
# displayed age flicker.
DEFAULT_DEMOGRAPHICS_REFRESH_QUALITY_RATIO = 3.0
# A hard ceiling on re-runs per person, so a camera that keeps producing
# slightly better crops cannot queue MiVOLO indefinitely.
DEFAULT_DEMOGRAPHICS_MAX_REFRESHES = 2
# Size of the square face crop as a multiple of the estimated head width.  A
# face detector's box runs from mid-forehead to chin; anchored on the nose, 1.5
# head widths covers that plus the margin MiVOLO's training crops carry.
DEFAULT_FACE_BOX_HEAD_WIDTH_SCALE = 1.5
