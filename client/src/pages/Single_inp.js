import React from 'react'
import axios from "axios";
import { useState, useEffect } from 'react';
import Header from '../components/Header';
import Navbar from '../components/Navbar';
import { className } from './Single_inp.css'

function Single_inp() {
    const [text, setText] = useState("");
    const [extract, setExtract] = useState("");
    const [actual, setActual] = useState("");
    const [abstract, setAbstract] = useState("");

    useEffect(() => {
        console.log("effect");
        console.log(abstract);
        console.log(extract);
        console.log(actual);
    }, [abstract, extract, actual])

    const handleInput = () => {
        const Actual_Summarizer = text;
        axios.post(
            `http://127.0.0.1:8000/single/`, { Actual_Summarizer })
            .then(res => {
                const d = res.data['Abstract_Summarizer'];
                const e = res.data['Extractive_Summarizer'];
                const a = res.data['Actual_Summarizer'];
                console.log(res.data);
                setAbstract(d);
                setExtract(e);
                setActual(a);
            })
    }
    return (
        <div>
            <Header />
            <Navbar />
            <div>
                {actual ? (
                    <>
                        <div>
                            <div className='box1'>
                            <div className='header'><h3>ACTUAL SUMMARY</h3></div>
                                <p>{actual}</p>
                            </div>
                            <div className='box1'>
                                <div className='header'><h3>EXTRACTIVE SUMMARY</h3></div>
                                <p>{extract} </p>
                            </div>
                            <div className='box1'>
                            <div className='header'><h3>ABSTRACTIVE SUMMARY</h3></div>
                                <p>{abstract}</p>
                            </div>
                        </div>
                    </>
                ) : (
                    <>
                        <div className='box'>
                            <p>Enter paragraph </p>
                            <textarea className='textbox' value={text} onChange={(e) => setText(e.target.value)} placeholder='Type here...'></textarea>
                            <button className='button' onClick={handleInput}>Submit</button>
                        </div>
                    </>
                )}
            </div>
        </div>
    )
}

export default Single_inp
