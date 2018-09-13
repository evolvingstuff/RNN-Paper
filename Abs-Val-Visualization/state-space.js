'use strict';

const WIDTH = 2000;
const HEIGHT = 500;
const UNIT = 450;
const OFFSET = 25;
const MAX_DEPTH = 10;
const NEG_INPUT = -0.1;

function render() {

	let alpha = parseFloat($('#inp_alpha').val())/1000;
	let beta = parseFloat($('#inp_beta').val())/1000;
	let rec = parseFloat($('#inp_rec').val())/1000;
	let flipped = document.getElementById('checkBox').checked;

	let ctx = document.getElementById('cnvs').getContext('2d');
	ctx.fillStyle = 'white';
	ctx.fillRect(0, 0, WIDTH, HEIGHT);

	extend(ctx, 0, alpha, beta, rec, 0, flipped);
}

function extend(ctx, loc, alpha, beta, rec, depth, flipped) {

	if (depth >= MAX_DEPTH) {
		return;
	}

	let from, to, r, x, y;

	//pos stroke
	ctx.beginPath();
	ctx.strokeStyle = 'rgba(0,0,0,0.2)';
	ctx.lineWidth = 1;
	from = Math.abs(loc)*UNIT;
	to = Math.abs(alpha + rec*loc)*UNIT;
	if (to < from) {
		let tmp = to;
		to = from;
		from = tmp;
	}
	r = Math.abs(from - to)/2;
	x = (from+to)/2 + OFFSET;
	y = HEIGHT/2;
	ctx.arc(x, y, r, Math.PI, 0);
	ctx.stroke();

	extend(ctx, Math.abs(alpha + rec*loc), alpha, beta, rec, depth+1, flipped);

	//neg stroke
	ctx.beginPath();
	ctx.strokeStyle = 'rgba(255,0,0,0.2)';
	ctx.lineWidth = 1;
	from = Math.abs(loc)*UNIT;
	to = Math.abs(beta + rec*loc)*UNIT;
	if (to < from) {
		let tmp = to;
		to = from;
		from = tmp;
	}
	r = Math.abs(from - to)/2;
	x = (from+to)/2 + OFFSET;
	y = HEIGHT/2;
	if (flipped) {
		ctx.arc(x, y, r, 0, Math.PI);
	}
	else {
		ctx.arc(x, y, r, Math.PI, 0);
	}
	ctx.stroke();

	extend(ctx, Math.abs(beta + rec*loc), alpha, beta, rec, depth+1, flipped);
}

function init() {

	$('#inp_alpha').on('input', render);
	$('#inp_beta').on('input', render);
	$('#inp_rec').on('input', render);
	$('#checkBox').on('change', render);

	render();
}