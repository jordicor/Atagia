-- Conversation activity is derived from one user's authoritative transcript.
-- It must never survive an ownership move or be written for a mismatched owner.

DELETE FROM conversation_activity_stats AS cas
WHERE NOT EXISTS (
    SELECT 1
    FROM conversations AS c
    WHERE c.id = cas.conversation_id
      AND c.user_id = cas.user_id
);

CREATE TRIGGER conversation_activity_owner_guard_bi
BEFORE INSERT ON conversation_activity_stats
WHEN NOT EXISTS (
    SELECT 1
    FROM conversations AS c
    WHERE c.id = NEW.conversation_id
      AND c.user_id = NEW.user_id
)
BEGIN
    SELECT RAISE(ABORT, 'conversation activity owner mismatch');
END;

CREATE TRIGGER conversation_activity_owner_guard_bu
BEFORE UPDATE ON conversation_activity_stats
WHEN NOT EXISTS (
    SELECT 1
    FROM conversations AS c
    WHERE c.id = NEW.conversation_id
      AND c.user_id = NEW.user_id
)
BEGIN
    SELECT RAISE(ABORT, 'conversation activity owner mismatch');
END;

CREATE TRIGGER conversation_activity_owner_move_cleanup_au
AFTER UPDATE OF user_id ON conversations
WHEN OLD.user_id IS NOT NEW.user_id
BEGIN
    DELETE FROM conversation_activity_stats
    WHERE conversation_id = NEW.id;
END;
