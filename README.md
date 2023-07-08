# Homogeneous-Multi-unit-Auction-over-a-Social-Network
This repo contains the implementation of the **VCG** [[1]](#1), **GIDM** [[2]](#2), **MUDAN** [[3]](#3), and **MUDAR** [[3]](#3) auctions for selling multiple homogeneous items on a social network, with each agent only requiring a single item. 

Each mechanism is implemented through a function `auction(k, seller_net, reports, bids)`, where:
- **k** is the number of item to sell;
- **seller_net** is a set of strings each identifying a different bidder;
- **reports** is a dictionary whose keys are strings each identifying a different bidder and whose values are sets of strings representing the set of bidders to which the bidder identified by the key reports the information about the auction;
- **bids** is a dictionary whose keys are strings each identifying a different bidder and whose values are numbers defining the bid of the bidder identified by that key.

The function returns two values:
- **allocation**, that is a dictionary that has as keys the strings identifying each of the biddersthat submitted a bid, and as value a boolean True if this bidder is allocated one of the items, and False otherwise.
- **payments**, that is a dictionary that has as keys the strings identifying each of the bidders that submitted a bid, and as value the price that she pays. Here, a positive price means that the bidder is paying to the seller, while a negative price means that the seller is paying to the bidder.


A more comlex use of these auctions can be found in the repo [bcirillo99/SocialNetworkAnalysis](https://github.com/bcirillo99/SocialNetworkAnalysis) where the goal of the project is to find in a social network, through the use of the bandit algorithm and multi-unit auction, at each step the node (seller) that maximizes total revenue (check the repo for more details).


## References
> [<a id="1">[1]</a> **Mechanism Design in Social Networks**](https://arxiv.org/abs/1702.03627)
> 
> Bin Li, Dong Hao, Dengji Zhao, Tao Zhou
> 
> *[arXiv:1702.03627](https://arxiv.org/abs/1702.03627)*
>
> [<a id="2">[2]</a> **Selling multiple items via social networks**](https://arxiv.org/abs/1903.02703)
> 
> Dengji Zhao, Bin Li, Junping Xu, Dong Hao, Nicholas R. Jennings
> 
> *[arXiv:1903.02703](https://arxiv.org/abs/1903.02703)*
>
> [<a id="3">[3]</a> **Multi-unit Auction over a Social Network**](https://arxiv.org/abs/2302.08924)
> 
> Yuan Fang, Mengxiao Zhang, Jiamou Liu, Bakh Khoussainov, Mingyu Xiao
> 
> *[arXiv:2302.08924](https://arxiv.org/abs/2302.08924)*
