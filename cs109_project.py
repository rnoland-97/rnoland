import numpy as np
import copy

from collections import namedtuple



def Create_RightDeck_LeftDeck(Right_Deck, Left_Deck):

	#Create a deck of cards
	Card = namedtuple('Card', ['value', 'suit'])
	suits = ['hearts', 'spades', 'diamonds', 'clubs']
	cards = [Card(value, suit) for value in range (1, 14) for suit in suits]


	List_Index_To_Split_Deck = np.random.choice(len(cards), 1, replace=True)
	Index_To_Split_Deck = List_Index_To_Split_Deck[0]


	#Build the Right Deck up to the Index_To_Split_Deck and Left Deck from remaining cards such that Index_To_Split_Deck number of cards are added to left deck and remaining cards added to left.
	counter = 0.0

	for i in range(len(cards)):
		if (counter < Index_To_Split_Deck):
			Right_Deck.append( (cards[i].value, cards[i].suit) )

			counter = counter + 1

		else:
			Left_Deck.append( (cards[i].value, cards[i].suit) )



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

	#Now shuffle Right and Left Decks 1-1 (ie a single card falls from Right Deck, followed by a single card from Left Deck falling on top of it)
	Shuffled_Deck = []


	Length_Of_Right_Deck = len(Right_Deck)
	Length_Of_Left_Deck = len(Left_Deck)

	#Keep track of the last index used from Right_Deck
	Right_Deck_Tracker = 0
	Left_Deck_Tracker = 0

	while Length_Of_Right_Deck >= 0:

		#Random Clump (i.e. the number of cards to add to clump)
		Right_Random_Clump = np.random.choice(5, 1, replace=True)

		#If the randomClump exceeds the number of cards left in the Right Deck, draw another randomClump
		while Right_Random_Clump - Length_Of_Right_Deck > 0:
			Right_Random_Clump = np.random.choice(5, 1, replace=True)

		#We need to account for taking randomClump number of cards out of the Right Deck into the shuffled deck
		Length_Of_Right_Deck = Length_Of_Right_Deck - Right_Random_Clump
			

		for i in range(len(Right_Random_Clump)):
			Card_From_Right_Deck = Right_Deck[i + int(Right_Deck_Tracker)]
			Shuffled_Deck.append(Card_From_Right_Deck)

		#Keep track of the last index used from Right_Deck
		Right_Deck_Tracker = Right_Deck_Tracker + Right_Random_Clump





		#Random Clump (i.e. the number of cards to add to clump)
		Left_Random_Clump = np.random.choice(5, 1, replace=True)

		#If the randomClump exceeds the number of cards left in the Left Deck, draw another randomClump
		while Left_Random_Clump - Length_Of_Left_Deck > 0:
			Left_Random_Clump = np.random.choice(5, 1, replace=True)

		#We need to account for taking randomClump number of cards out of the Left Deck into the shuffled deck
		Length_Of_Left_Deck = Length_Of_Left_Deck - Left_Random_Clump


		for i in range(len(Left_Random_Clump)):
			Card_From_Left_Deck = Left_Deck[i + int(Left_Deck_Tracker)]
			Shuffled_Deck.append(Card_From_Left_Deck)

		#Keep track of the last index used from Left_Deck
		Left_Deck_Tracker = Left_Deck_Tracker + Left_Random_Clump










	





	
	







def main():

	#"Grab half cards" by picking uniformly a random index among all indices 0-51
	Right_Deck = []
	Left_Deck = []

	print("****************************************************")
	Create_RightDeck_LeftDeck(Right_Deck, Left_Deck)
	print("****************************************************")



	print("****************************************************")
	#First card placed down is from Right Deck
	#print("\tReturn value was:", Shuffle_Without_Clumps(Right_Deck, Left_Deck))

	#First card placed down is from Left Deck
	#print("\tReturn value was:", Shuffle_Without_Clumps(Left_Deck, Right_Deck))
	print("****************************************************")

	print("****************************************************")
	print("\tReturn value was:", Shuffle_With_Clumps(Right_Deck, Left_Deck))
	#print("\tReturn value was:", optional_function('background.csv', 'learningOutcomes.csv'))
	print("****************************************************")


	print("Done!")


# This if-condition is True if this file was executed directly.
# It's False if this file was executed indirectly, e.g. as part
# of an import statement.
if __name__ == "__main__":
	main()