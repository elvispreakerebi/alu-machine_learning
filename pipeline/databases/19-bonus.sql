-- Stored procedure: AddBonus(user_id, project_name, score)
-- Creates project if not exists, then adds correction
DELIMITER $$
CREATE PROCEDURE AddBonus(IN user_id INT, IN project_name VARCHAR(255), IN score INT)
BEGIN
    IF (SELECT COUNT(*) FROM projects WHERE name = project_name) = 0 THEN
        INSERT INTO projects (name) VALUES (project_name);
    END IF;
    INSERT INTO corrections (user_id, project_id, score)
    VALUES (user_id, (SELECT id FROM projects WHERE name = project_name LIMIT 1), score);
END$$
DELIMITER ;
