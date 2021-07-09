Params = {
    'ActionPenalty': 0,
    'PenalisedActions': { Actions.Left, Actions.Right, Actions.RotateLeft, Actions.RotateRight },
    'GameOver': -30,
    'ScoreScaling': 4,
    'PieceLanded': 2,
    'HoleCreated': -5,
    'MaxHeightChange': (lambda old_height, new_height: 0 )      # Disable max height reward
#        'MaxHeightChange': (lambda old_height, new_height: -max(0,new_height-old_height)**1.5 )
}


def action_taken(action):
    if action in Params['PenalisedActions']:
        return Params['ActionPenalty']

    return 0


def line_cleared(points_scored):
    return points_scored*self.RewardInfo['ScoreScaling']


def piece_landed(board):
    reward = 0
    reward += Params['PieceLanded']

    holes_created = self.board.count_holes() - self.num_holes
    reward += holes_created * Params['HoleCreated']

    new_max_height = self.board.max_height()
    reward += Params['MaxHeightChange'](self.max_height, new_max_height)

    return reward


def game_over():
    return Params['GameOver']
