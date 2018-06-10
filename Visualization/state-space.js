'use strict';

const WIDTH = 2000;
const HEIGHT = 500;
const UNIT = 450;
const OFFSET = 25;
const MAX_DEPTH = 10;

function render() {

	let alpha = parseFloat($('#inp_x').val());
	let beta = parseFloat($('#inp_y').val())/1000;

	let ctx = document.getElementById('cnvs').getContext('2d');
	ctx.fillStyle = 'white';
	ctx.fillRect(0, 0, WIDTH, HEIGHT);

	extend(ctx, 0, alpha, beta, 0);
}

function extend(ctx, loc, alpha, beta, depth) {

	if (depth >= MAX_DEPTH) {
		return;
	}

	let from, to, r, x, y;

	//pos stroke
	ctx.beginPath();
	ctx.strokeStyle = 'rgba(0,0,0,0.2)';
	ctx.lineWidth = 1;
	from = Math.abs(loc)*UNIT;
	to = Math.abs(alpha + beta*loc)*UNIT;
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

	extend(ctx, Math.abs(alpha+beta*loc), alpha, beta, depth+1);

	//neg stroke
	ctx.beginPath();
	ctx.strokeStyle = 'rgba(255,0,0,0.2)';
	ctx.lineWidth = 1;
	from = Math.abs(loc)*UNIT;
	to = Math.abs(beta*loc)*UNIT;
	if (to < from) {
		let tmp = to;
		to = from;
		from = tmp;
	}
	r = Math.abs(from - to)/2;
	x = (from+to)/2 + OFFSET;
	y = HEIGHT/2;
	//ctx.arc(x, y, r, Math.PI, 0);
	ctx.arc(x, y, r, 0, Math.PI);
	ctx.stroke();

	extend(ctx, Math.abs(beta*loc), alpha, beta, depth+1);
}

function init() {

	$('#inp_y').on('input', render);

	render();
}