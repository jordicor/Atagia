-- Close cross-conversation and ownership-move gaps while a user's canonical
-- transcript is being rebuilt. Both sides of an UPDATE are authoritative.

DROP TRIGGER transcript_rebuild_block_message_update;

CREATE TRIGGER transcript_rebuild_block_message_update
BEFORE UPDATE ON messages
WHEN EXISTS (
    SELECT 1
    FROM conversations AS c
    JOIN conversation_transcript_selections AS selection
      ON selection.user_id = c.user_id
    WHERE c.id IN (OLD.conversation_id, NEW.conversation_id)
      AND selection.state IN ('rebuilding', 'remediation_required')
)
BEGIN
    SELECT RAISE(ABORT, 'selected transcript rebuild blocks user message writes');
END;

CREATE TRIGGER transcript_rebuild_block_conversation_owner_update
BEFORE UPDATE OF user_id ON conversations
WHEN OLD.user_id IS NOT NEW.user_id
 AND EXISTS (
    SELECT 1
    FROM conversation_transcript_selections AS selection
    WHERE selection.user_id IN (OLD.user_id, NEW.user_id)
      AND selection.state IN ('rebuilding', 'remediation_required')
)
BEGIN
    SELECT RAISE(ABORT, 'selected transcript rebuild blocks conversation ownership moves');
END;
