import numpy as np
import copy
import math

from collections import namedtuple



def Create_RightDeck_LeftDeck(Right_Deck, Left_Deck):

	#Create a deck of cards
	Card = namedtuple('Card', ['value', 'suit'])
	suits = ['hearts', 'spades', 'diamonds', 'clubs']
	cards = [Card(value, suit) for value in range (1, 14) for suit in suits]


	# Get a random Index to split the deck
	'''List_Index_To_Split_Deck = np.random.choice(len(cards), 1, replace=True)
	Index_To_Split_Deck = List_Index_To_Split_Deck[0]
	'''


	#Implement the Binomial Distribution to get accurate split. It turns out that a deck of cards is cut into two portiosn according to a binomial distribution of (N choose K) / 2^N where N = number of cards in deck, K = number cards cut off into left deck
	#first, create a list of all the probabilites for each k
	List_Of_Index_K = []
	for i in range(0, 53):
		List_Of_Index_K.append(i)

	#second, create a list for each probability that k cards are sorted into sub-deck (1 of two split-decks) 
	List_Of_Probability_K = []

	f = math.factorial
	for k in range(0,53): 

		#According to the paper- "the chance that k cards are cut off is (n choose k)/ 2^n for 0 =< k =< n" 
		Probability_k = ( (f(52) / f(k) / f(52-k)) / 2**(52) )
		List_Of_Probability_K.append(Probability_k)

	#Get an Index to split the deck according to the binomial distribution 
	List_Index_To_Split_Deck = np.random.choice(List_Of_Index_K, 1, replace=True, p= List_Of_Probability_K)
	Index_To_Split_Deck = List_Index_To_Split_Deck[0]


	counter = 0.0

	for i in range(len(cards)):
		if (counter < Index_To_Split_Deck):
			Right_Deck.append( (i) )

			counter = counter + 1

		else:
			Left_Deck.append( (i) )

	


	'''#Build the Right Deck up to the Index_To_Split_Deck and Left Deck from remaining cards such that Index_To_Split_Deck number of cards are added to left deck and remaining cards added to left.
	counter = 0.0

	for i in range(len(cards)):
		if (counter < Index_To_Split_Deck):
			Right_Deck.append( (cards[i].value, cards[i].suit) )

			counter = counter + 1

		else:
			Left_Deck.append( (cards[i].value, cards[i].suit) )

'''

def Shuffle_Without_Clumps(Right_Deck, Left_Deck):

	#Now shuffle Right and Left Decks 1-1 (ie a single card falls from Right Deck, followed by a single card from Left Deck falling on top of it)
	Shuffled_Deck = []

	#If the Right deck is bigger than Left, go through all the right first 
	if ( len(Right_Deck) > len(Left_Deck) ):

		print('Right bigger than left')

		for i in range(len(Right_Deck)):
				#Add card from Right Deck to Shuffled Deck
			Card_From_Right_Deck = Right_Deck[i]
			Shuffled_Deck.append(Card_From_Right_Deck)

				#Add card from Left Deck to Shuffled Deck BUT ONLY IF LEFT_DECK is not empty
			if  len(Left_Deck) != 0 and len(Left_Deck) >= i+1:
			
				Card_From_Left_Deck = Left_Deck[i]
				Shuffled_Deck.append(Card_From_Left_Deck)

		print(len(Shuffled_Deck))

	#If the Left deck is bigger, go through all left adding in 
	elif (len(Left_Deck) > len(Right_Deck)):

		print('Left bigger than right')

		for i in range(len(Right_Deck)):
				#Add card from Right Deck to Shuffled Deck
			Card_From_Right_Deck = Right_Deck[i]
			Shuffled_Deck.append(Card_From_Right_Deck)

				#Add card from Left Deck to Shuffled Deck BUT ONLY IF LEFT_DECK is not empty
			if len(Left_Deck) != 0 and len(Left_Deck) >= i+1:
			
				Card_From_Left_Deck = Left_Deck[i]
				Shuffled_Deck.append(Card_From_Left_Deck)

			#if there are still elements in the left deck, add them to end of Shuffled deck
		if len(Left_Deck) != 0:
			for i in range(len(Left_Deck) - len(Right_Deck)):
				Shuffled_Deck.append(Left_Deck[i + len(Right_Deck)] )


		print(len(Shuffled_Deck))

	#If the Left Deck and Right Deck are the same size
	elif (len(Left_Deck) == len(Right_Deck)):

		print('Right same as = left')

		for i in range(len(Right_Deck)):
			Card_From_Right_Deck = Right_Deck[i]
			Shuffled_Deck.append(Card_From_Right_Deck)

			Card_From_Left_Deck = Left_Deck[i]
			Shuffled_Deck.append(Card_From_Left_Deck)

		print(len(Shuffled_Deck))

	print(Shuffled_Deck)







def Shuffle_With_Clumps(Right_Deck, Left_Deck):

	from collections import deque
	
	#Now shuffle Right and Left Decks 1-1 (ie a single card falls from Right Deck, followed by a single card from Left Deck falling on top of it)
	Shuffled_Deck = []

	Right_Deck_Queue = deque()
	Left_Deck_Queue = deque()

	#Creat a queue for Right Deck
	for i in range(len(Right_Deck)):
		Right_Deck_Queue.append(Right_Deck[i])

	#Creat a queue for Left Deck
	for i in range(len(Left_Deck)):
		Left_Deck_Queue.append(Left_Deck[i])



	print(Right_Deck_Queue)
	print('OAHSDIFHAOSIDHF')
	print(Left_Deck_Queue)

	if ( len(Right_Deck) > len(Left_Deck) ):

		print('Right bigger than left')


		#Add card from Right Deck to Shuffled Deck
		for i in range(len(Right_Deck_Queue)):

			#Get a clump of Right Cards
			Right_Random_Clump_List = np.random.choice(5, 1, replace=True)
			Right_Random_Clump_Number = Right_Random_Clump_List[0]


			#If the Clump of right cards exceeds the number of cards remaining in the Right Deck, draw another Clump of Right cards
			while  len(Right_Deck_Queue) - Right_Random_Clump_Number < 0:
				Right_Random_Clump_List = np.random.choice(5, 1, replace=True)
				Right_Random_Clump_Number = Right_Random_Clump_List[0]


			#Add Clump of Right cards to Shuffled Deck
			for j in range((Right_Random_Clump_Number)):
				Shuffled_Deck.append(Right_Deck_Queue.popleft())



			#Add card from Left Deck to Shuffled Deck BUT ONLY IF LEFT_DECK is not empty
			if  len(Left_Deck) != 0 and len(Left_Deck) >= i+1:
			
				#Get a clump of Left Cards
				Left_Random_Clump_List = np.random.choice(5, 1, replace=True)
				Left_Random_Clump_Number = Left_Random_Clump_List[0]

				#If the Clump of left cards exceeds the number of cards remaining in the Left Deck, draw another Clump of Left cards
				while  len(Left_Deck_Queue) - Left_Random_Clump_Number < 0:
					Left_Random_Clump_List = np.random.choice(5, 1, replace=True)
					Left_Random_Clump_Number = Left_Random_Clump_List[0]

				#Add Clump of Left cards to Shuffled Deck
				for j in range((Left_Random_Clump_Number)):
					Shuffled_Deck.append(Left_Deck_Queue.popleft())


	elif (len(Left_Deck) > len(Right_Deck)):

		print('Left bigger than right')


		#Add card from Right Deck to Shuffled Deck
		for i in range(len(Right_Deck_Queue)):

			#Get a clump of Right Cards
			Right_Random_Clump_List = np.random.choice(5, 1, replace=True)
			Right_Random_Clump_Number = Right_Random_Clump_List[0]


			#If the Clump of right cards exceeds the number of cards remaining in the Right Deck, draw another Clump of Right cards
			while  len(Right_Deck_Queue) - Right_Random_Clump_Number < 0:
				Right_Random_Clump_List = np.random.choice(5, 1, replace=True)
				Right_Random_Clump_Number = Right_Random_Clump_List[0]


			#Add Clump of Right cards to Shuffled Deck
			for j in range((Right_Random_Clump_Number)):
				Shuffled_Deck.append(Right_Deck_Queue.popleft())



			#Add card from Left Deck to Shuffled Deck BUT ONLY IF LEFT_DECK is not empty
			if  len(Left_Deck) != 0 and len(Left_Deck) >= i+1:
			
				#Get a clump of Left Cards
				Left_Random_Clump_List = np.random.choice(5, 1, replace=True)
				Left_Random_Clump_Number = Left_Random_Clump_List[0]

				#If the Clump of left cards exceeds the number of cards remaining in the Left Deck, draw another Clump of Left cards
				while  len(Left_Deck_Queue) - Left_Random_Clump_Number < 0:
					Left_Random_Clump_List = np.random.choice(5, 1, replace=True)
					Left_Random_Clump_Number = Left_Random_Clump_List[0]

				#Add Clump of Left cards to Shuffled Deck
				for j in range((Left_Random_Clump_Number)):
					Shuffled_Deck.append(Left_Deck_Queue.popleft())


		#if there are still elements in the left deck, add them to end of Shuffled deck
		if len(Left_Deck_Queue) != 0:

			#Add remaining cards to Shuffled Deck
			for i in range(len(Left_Deck_Queue)):
				Shuffled_Deck.append(Left_Deck_Queue.popleft() )


	#If the Left Deck and Right Deck are the same size
	elif (len(Left_Deck) == len(Right_Deck)):

		print('Right same as = left')

		for i in range(len(Right_Deck_Queue)):

			#Get a clump of Right Cards
			Right_Random_Clump_List = np.random.choice(5, 1, replace=True)
			Right_Random_Clump_Number = Right_Random_Clump_List[0]

			#If the Clump of right cards exceeds the number of cards remaining in the Right Deck, draw another Clump of Right cards
			while  len(Right_Deck_Queue) - Right_Random_Clump_Number < 0:
				Right_Random_Clump_List = np.random.choice(5, 1, replace=True)
				Right_Random_Clump_Number = Right_Random_Clump_List[0]


			#Add Clump of Right cards to Shuffled Deck
			for j in range((Right_Random_Clump_Number)):
				Shuffled_Deck.append(Right_Deck_Queue.popleft())


			#Add card from Left Deck to Shuffled Deck BUT ONLY IF LEFT_DECK is not empty
			if  len(Left_Deck) != 0 and len(Left_Deck) >= i+1:
			
				#Get a clump of Left Cards
				Left_Random_Clump_List = np.random.choice(5, 1, replace=True)
				Left_Random_Clump_Number = Left_Random_Clump_List[0]

				#If the Clump of left cards exceeds the number of cards remaining in the Left Deck, draw another Clump of Left cards
				while  len(Left_Deck_Queue) - Left_Random_Clump_Number < 0:
					Left_Random_Clump_List = np.random.choice(5, 1, replace=True)
					Left_Random_Clump_Number = Left_Random_Clump_List[0]

				#Add Clump of Left cards to Shuffled Deck
				for j in range((Left_Random_Clump_Number)):
					Shuffled_Deck.append(Left_Deck_Queue.popleft())


	print(len(Shuffled_Deck))
	print((Shuffled_Deck))



def Shuffle_According_To_Paper(Right_Deck, Left_Deck):
	from collections import deque
	from numpy.random import rand

	
	#Now shuffle Right and Left Decks 1-1 (ie a single card falls from Right Deck, followed by a single card from Left Deck falling on top of it)
	Shuffled_Deck = []

	Right_Deck_Queue = deque()
	Left_Deck_Queue = deque()

	#Creat a queue for Right Deck
	for i in range(len(Right_Deck)):
		Right_Deck_Queue.append(Right_Deck[i])

	#Creat a queue for Left Deck
	for i in range(len(Left_Deck)):
		Left_Deck_Queue.append(Left_Deck[i])


	ValueA = len(Left_Deck)
	ValueB = len(Right_Deck)

	#While there are still cards in either the right or left decks
	while ((ValueA + ValueB) != 0 ):
		
		#probability that card drops from left deck is ValueA / (ValueA + ValueB)
		if rand() < (ValueA / (ValueA + ValueB) ):
			Prob_Drop_Card_From_Left = 1
		else :
			Prob_Drop_Card_From_Left = 0 

		#print(Prob_Drop_Card_From_Left)

		#If the the card drops from the left deck, then add it to the Shuffled_Deck and remove one from ValueA, else do same for right deck
		if Prob_Drop_Card_From_Left == 1:
			Shuffled_Deck.append(Left_Deck_Queue.popleft())
			ValueA = ValueA - 1

			#print(ValueA)
			#print(Shuffled_Deck)

		else:
			Shuffled_Deck.append(Right_Deck_Queue.popleft())
			ValueB = ValueB - 1

			#print(ValueB)
			#print(Shuffled_Deck)

	#print(len(Shuffled_Deck))

	return (Shuffled_Deck)

	
def is_in_bounds(pos_y, pos_x, x_min, x_max, y_min, y_max):
	#Check that new pos in bounds
	if pos_x >= x_min and pos_x <= x_max and pos_y >= y_min and pos_y <= y_max:
		return True
	else:
		return False

	
	
def Shuffle_With_Shmoosh():
	import random
	from collections import deque
	from numpy.random import rand

	#We just want the single deck (Not a Right and Left Deck)
	Deck = deque()
	for i in range(52):
			Deck.append(i)

	#It takes about 1 second to move cards from one cell in matrix to another cell
	#This was verified by doing so with a deck of cards on a table many times
	#Therefore, we can expect that it takes about 1 second to move some number n of cards from one position in matrix Square_Plane to another 
	#Note! I have multipled Time_In_Seconds by Number_Of_Hands_Shuffling_Deck to account for two hands shuffling deck.
	#The intuition is that shmoosh shuffling with two hands is equivalent to shoosh shuffling with one hand for twice as long 
	Number_Of_Hands_Shuffling_Deck = 2
	Seconds_Spent_Shuffling = 60
	Time_in_Seconds = Seconds_Spent_Shuffling * Number_Of_Hands_Shuffling_Deck

	#Row Length
	Row_Length = 3

	#Column_Length 
	Column_Length = 3

	#Create the rectangular plane where we will be smooshing cards.
	#This matrix represents a 3x3 Square table where we will be smooshing. 
	#The Deck of 52 cards are initially in the center (location 3x3) of the Square table 
	Square_Plane = np.array([ [0,0,0],[0,Deck,0],[0,0,0] ])

	for row in range(Row_Length):
		for column in range(Column_Length):
			Square_Plane[row][column] = deque()
	Square_Plane[1][1] = (Deck)

	#Keep track of position of "Hand" as it moves through the matrix 
	#starting position set at 1x1 for the 3x3 matrix 
	X_Axis_Move = 1
	Y_Axis_Move = 1

	#Keep track of position of "Hand" to ensure that it is not outside bounds of Square_Plane
	pos = [1, 1]

	#Min and Max x-axis values according to Square_Plane matrix
	x_min = 0
	x_max = 2

	#Min and Max y-axis values according to Square_Plane matrix
	y_min = 0
	y_max = 2

	#Hand can only move either +1 square X-axid, -1 square X-axis, 1 square Y-Axis, -1 square Y-Axis
	#Want to uniformly pick a value of [0][1], [0][-1], [1][0], [-1][0]
	Pick_Position_List = []

	Pick_Position_List.append((0,1))
	Pick_Position_List.append((0,-1))
	Pick_Position_List.append((1,0) )
	Pick_Position_List.append((-1,0))



	for i in range(Time_in_Seconds):

		#Move Hand uniformly in some direction of the matrix of either up, or down by 1 
		#Choose a random value of either 1Up, -1Up, 1Down, -1Down
		#The values chosen are from the list Pick_Position_List which was created just before for loop
		Move_Hand_Value = Pick_Position_List[random.randint(0,3)]
		
		print('Move Hand Value')
		print(Move_Hand_Value)

		#If move (0,1) AND CHECK THAT STILL IN BOUNDS OF MATRIX
		if Move_Hand_Value[0] == 0 and Move_Hand_Value[1] == 1:

			#Check That new position is still in bounds of matrix
			pos[1] += 1

			if is_in_bounds(pos[0], pos[1], x_min, x_max, y_min, y_max): 

				#for every card at current position (Added list(Square_Plane[Y_Axis_Move][X_Axis_Move]) to avoid changing deque while iterating
				for card in list(Square_Plane[Y_Axis_Move][X_Axis_Move]):

					#Using Bernoulli, if heads then move card to new position
					if rand() < 0.5:

						#Add current card to new Position
						Square_Plane[Y_Axis_Move][X_Axis_Move + 1].append(card)

						#Delete current card from old Position 
						Square_Plane[Y_Axis_Move][X_Axis_Move].remove(card)

				#Update position of Hand 
				X_Axis_Move += 1

				print(Y_Axis_Move, X_Axis_Move)

			#Account for possibility that we do go out bounds, so then take away the 1 value added to pos before for loop
			else:
				pos[1] = pos[1] - 1
			


		#If move (0,-1) AND CHECK THAT STILL IN BOUNDS OF MATRIX
		elif Move_Hand_Value[0] == 0 and Move_Hand_Value[1] == -1:

			#Check That new position is still in bounds of matrix
			pos[1] = pos[1] - 1

			if is_in_bounds(pos[0],pos[1], x_min, x_max, y_min, y_max):

				#for every card at current position (Added list(Square_Plane[Y_Axis_Move][X_Axis_Move]) to avoid changing deque while iterating
				for card in list(Square_Plane[Y_Axis_Move][X_Axis_Move]):

					#Using Bernoulli, if heads then move card to new position
					if rand() < 0.5:

						#Add current card to new Position
						Square_Plane[Y_Axis_Move][X_Axis_Move - 1].append(card)

						#Delete current card from old Position 
						Square_Plane[Y_Axis_Move][X_Axis_Move].remove(card)

				#Acount for new position of Hand 
				X_Axis_Move -= 1

				print(Y_Axis_Move, X_Axis_Move)

			#Account for possibility that we do go out bounds, so then take away the 1 value added to pos
			else:
				pos[1] = pos[1] + 1
			

		#If move (1,0) AND CHECK THAT STILL IN BOUNDS OF MATRIX
		elif Move_Hand_Value[0] == 1 and Move_Hand_Value[1] == 0:

			#Check That new position is still in bounds of matrix
			pos[0] = pos[0] + 1
			if is_in_bounds(pos[0], pos[1], x_min, x_max, y_min, y_max):

				#for every card at current position (Added list(Square_Plane[Y_Axis_Move][X_Axis_Move]) to avoid changing deque while iterating 
				for card in list(Square_Plane[Y_Axis_Move][X_Axis_Move]):

					#Using Bernoulli, if heads then move card to new position
					if rand() < 0.5:

						#Add current card to new Position
						Square_Plane[Y_Axis_Move + 1][X_Axis_Move].append(card)

						#Delete current card from old Position 
						Square_Plane[Y_Axis_Move][X_Axis_Move].remove(card)

				#Acount for new position of Hand 
				Y_Axis_Move += 1

				print(Y_Axis_Move, X_Axis_Move)

			#Update Pos (Our check to make sure we never go out of bounds of matrix)
			else:
				pos[0]=pos[0] - 1
			

		#If move (-1,0) AND CHECK THAT STILL IN BOUNDS OF MATRIX
		elif Move_Hand_Value[0] == -1 and Move_Hand_Value[1] == 0:

			pos[0] = pos[0] - 1
			if is_in_bounds(pos[0], pos[1], x_min, x_max, y_min, y_max):

				#for every card at current position (Added list(Square_Plane[Y_Axis_Move][X_Axis_Move]) to avoid changing deque while iterating 
				for card in list(Square_Plane[Y_Axis_Move][X_Axis_Move]):

					#Using Bernoulli, if heads then move card to new position
					if rand() < 0.5:

						#Add current card to new Position
						Square_Plane[Y_Axis_Move - 1][X_Axis_Move].append(card)

						#Delete current card from old Position 
						Square_Plane[Y_Axis_Move][X_Axis_Move].remove(card)

				#Acount for new position of Hand 
				Y_Axis_Move -= 1

				print(Y_Axis_Move, X_Axis_Move)

			#Update Pos (Our check to make sure we never go out of bounds of matrix)
			else:
				pos[0]=pos[0] + 1

		print('*****************')
		print(Square_Plane)


	#Go through each every row in each column (So every point in Y-Axis for each X-Axis) and add all cards into an independent list at each X-Axis point
	#Concatenate together these independent lists of cards along the X-Axis to create our shuffled deck 
	#In other words, the cards are arranged linearly along the X-Axis in an increasing order that respects the X-Axis values 
	Shuffled_Deck_Smoosh = []

	#For each X-Axis in Square_Plane
	for column in range(Column_Length):

		#For each Y-Axis in Square_Plane
		for row in range(Row_Length):

			#For every card in this cell of Square_Plane- Add it to shuffled deck 
			for card in list(Square_Plane[row][column]):
				Shuffled_Deck_Smoosh.append(card)

			#Initialize this cell in Matrix Square_Plane to be an empty deque
			Square_Plane[row][column] = deque()

	print('*****************')
			
	print(Shuffled_Deck_Smoosh)
	print(len(Shuffled_Deck_Smoosh))

		





def main():

	#"Grab half cards" by picking uniformly a random index among all indices 0-51
	Right_Deck = []
	Left_Deck = []

	print("****************************************************")
	Create_RightDeck_LeftDeck(Right_Deck, Left_Deck)
	print("****************************************************")



	print("****************************************************")
	#First card placed down is from Right Deck
	#print("\tReturn value was:", Shuffle_Without_Clumps(Right_Deck, Left_Deck) )
	print("****************************************************")



	print("****************************************************")
	#First card placed down is from Left Deck
	#print("\tReturn value was:", Shuffle_Without_Clumps(Left_Deck, Right_Deck))
	print("****************************************************")



	print("****************************************************")
	#First card placed down is from Right Deck
	#print("\tReturn value was:", Shuffle_With_Clumps(Right_Deck, Left_Deck))
	print("****************************************************")


	print("****************************************************")
	#First card placed down is from Left Deck
	#print("\tReturn value was:", Shuffle_With_Clumps(Left_Deck, Right_Deck))
	print("****************************************************")


	print("****************************************************")
	#First card placed down is from Left Deck
	print("\tReturn value was:", Shuffle_With_Shmoosh() )
	print("****************************************************")

	print("****************************************************")
	#First card placed down is from Left Deck



	'''
	appearances = {} 

	for i in range(2000):

		#Choose one of the Shuffle Techniques with which to shuffle the cards
		#Each call of a shuffle function only shuffles the cards once according to that model.
		#To Shuffle the cards multiple times according to a shuffle function, pls call the shuffle function desired number of times

		ShuffledDeck = Shuffle_According_To_Paper(Right_Deck, Left_Deck)
		#ShuffledDeck = Shuffle_With_Clumps(Right_Deck, Left_Deck)
		#ShuffledDeck = Shuffle_Without_Clumps(Right_Deck, Left_Deck)


		Immutable_ShuffledDeck = ','.join([str(v) for v in ShuffledDeck])

		if Immutable_ShuffledDeck in appearances:
			appearances[Immutable_ShuffledDeck] += 1 
		else:
			appearances[Immutable_ShuffledDeck] = 1


	import matplotlib.pyplot as plt


	dictionary = appearances
	plt.bar(list(dictionary.keys()), dictionary.values(), color='g')
	plt.show()


	print("****************************************************")
	#print("\tReturn value was:", Shuffle_Histogram_Maker(Shuffled_Deck)
	print("****************************************************")


	print("Done!")
'''


# This if-condition is True if this file was executed directly.
# It's False if this file was executed indirectly, e.g. as part
# of an import statement.
if __name__ == "__main__":
	main()