import React from 'react'

import Header from '../components/Header'
import Navbar from '../components/Navbar'
import { className } from './Home.css'

function Home() {
    return (
        <div>
            <Header />
            <Navbar />
            <div className='b'>
                <h3 className='h3'>
                    SINGLE DOCUMENT SUMMARIZATION
                </h3>
                <p>
                Single‐document summarization transforms a source text into a condensed, shorter text in which the relevant information is preserved. The research into abstracts and informative extracts conducted by Earl at Lockheed Missiles and Space Co. studied the role of morphology, phonetics and syntax in summaries.
                </p>
            </div>
            <div className='b'>
                <h3>
                    What is extractive summarization?
                </h3>
                <p>
                    Extractive summarization is basically creating a summary based on strictly what you get in the text. It can be compared to copying down the main points of a text without any modification to those points and rearranging the order of that points and the grammar to make more sense out of the summary.
                 
                    Here we used textrank algorithm.

                    TextRank is a graph based algorithm for Natural Language Processing that can be used for keyword and sentence extraction. The algorithm is inspired by PageRank to rank sentences based similar matrics.    
                </p>
            </div>
            <div className='b'>
                <h3>
                    What is abstractive summarization?
                </h3>
                <p>
                    Abstractive Text Summarization is the task of generating a short and concise summary that captures the salient ideas of the source text. The generated summaries potentially contain new phrases and sentences that may not appear in the source text.

                    We use the same article to summarize as before, but this time, we use a transformer model from Huggingface
                </p>
            </div>
        </div>
    )
}

export default Home
