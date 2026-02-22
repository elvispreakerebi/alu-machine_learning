-- Stored procedure: ComputeAverageScoreForUser(user_id)
-- Computes and stores average score from corrections into users.average_score
DELIMITER $$
CREATE PROCEDURE ComputeAverageScoreForUser(IN user_id INT)
BEGIN
    UPDATE users
    SET average_score = (SELECT IFNULL(AVG(score), 0) FROM corrections WHERE corrections.user_id = user_id)
    WHERE users.id = user_id;
END$$
DELIMITER ;
