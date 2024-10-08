import random

import ChessRules
import ChessRules
import pygame as p
import numpy as np
import torch
import math
import torch.nn as nn
import torch.functional as F

class Node:
    def __init__(self, game, args, state, whiteKingLocation, blackKingLocation, isPossibleEnpassant, castlingRights, parent=None, action_taken=None, prior=0, visit_count=0):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior

        self.children = []
        # self.expandable_moves = game.get_valid_moves(state), removing this, but actually could be used to make search phase more effficient
        self.visit_count = visit_count
        self.value_sum = 0

        # Keeping track of kings' placement to make legal move generation and castling more simple
        self.whiteKingLocation = whiteKingLocation
        self.blackKingLocation = blackKingLocation

        # En passant
        self.isPossibleEnpassant = isPossibleEnpassant   # Tile where en passant capture can happen
        self.enPassantLogs = [self.isPossibleEnpassant]

        # Castling
        self.currentCastlingRights = castlingRights
        self.castleRightsLog = [ChessRules.CastleRights(self.currentCastlingRights.wKs, self.currentCastlingRights.wQs,
                                             self.currentCastlingRights.bKs, self.currentCastlingRights.bQs)]

    def is_fully_expanded(self):
        return len(self.children) > 0

    def select(self):
        best_child = None
        best_ucb = -np.inf

        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb
        return best_child

    def get_ucb(self, child):
        if child.visit_count == 0:
            q_value = 0
        else:
            q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        return q_value + self.args['C'] * (math.sqrt(self.visit_count) / (1 + child.visit_count)) * child.prior

    def expand(self, policy, validMoves):
        for action, prob in enumerate(policy):
            if prob > 0:
                child_state = self.state.copy()
                child_state = self.game.makeMoveAZ(child_state, validMoves[action], 1)
                child_state = self.game.changePerspective(child_state, player=-1)

                child = Node(self.game, self.args, child_state, self, action, prob)
                self.children.append(child)

        return child

    def simulate(self):
        value, is_terminal = self.game.get_value_and_terminate(self.state, self.action_taken)
        value = self.game.getOpponentValue(value)
        if is_terminal:
            return value

        rollout_state = self.state.copy()
        rollout_player = 1
        while True:
            valid_moves = self.game.getValidMoves(rollout_state)
            action = np.random.choice(np.where(valid_moves == 1)[0])
            rollout_state = self.game.get_next_state(rollout_state, action, rollout_player)
            value, is_terminal = self.game.get_value_and_terminate(rollout_state, action)
            if is_terminal:
                if rollout_player == -1:
                    value = self.game.getOpponentValue(value)
                return value
            rollout_player = self.game.getOpponent(rollout_player)

    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1

        value = self.game.get_opponent_value(value)
        if self.parent is not None:
            self.parent.backpropagate(value)

class MCTS:
    def __init__(self, args, state, player, game, model):
        self.args = args
        self.state = state
        self.player = player
        self.game = game
        self.model = model

    @torch.no_grad()
    def search(self, state, model):
        root = Node(self.game, self.args, state, visit_count=1)
        policy, _ = model(
            torch.tensor(self.game.get_encoded_state(state), device=model.device).unsqueeze(0)
        )
        policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
        policy = ((1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon']
                  * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size))

        validMoves = self.game.getValidMovesAZ(state)
        policy *= validMoves

        policy /= np.sum(policy)
        root.expand(policy)

        for search in range(self.args['num_searches']):
            node = root

            while node.is_fully_expanded():
                node = node.select()

            value, is_terminal = self.game.getValue_and_is_terminated(node.state, node.action_taken)
            value = self.game.getOpponentValue(value)

            if not is_terminal:
                policy, value = model(
                    torch.tensor(self.game.get_encoded_state(node.state), device=model.device).unsqueeze(0)
                )
                policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
                valid_moves = self.game.getValidMoves(node.state)
                policy *= valid_moves
                policy /= np.sum(policy)

                value = value.item()

                node.expand(policy)

            node.backpropagate(value)

        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs

    def findMove(self, state, validMoves):
        action_probs = self.search(state, self.model)
        bestMove = np.argmax(action_probs)
        chessMove = validMoves[bestMove]
        return chessMove

class ResNet(nn.Module):
    def __init__(self, game, num_resBlocks, num_hidden, device):
        super().__init__()
        self.device = device
        self.startBlock = nn.Sequential(
            nn.Conv2d(3, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )
        self.backBone = nn.ModuleList(
            [ResBlock(num_hidden) for i in range(num_resBlocks)]
        )
        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * game.rowCount * game.columnCount, game.actionSize)
        )

        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * game.rowCount * game.columnCount, 1),
            nn.Tanh()
        )
        self.to(device)

    def forward(self, x):
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return policy,

class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x


