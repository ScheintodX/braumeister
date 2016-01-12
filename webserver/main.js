#!/usr/bin/nodejs

require( './polyfill.js' );
require( 'colors' );
var util = require( 'util' );
var async = require( 'async' );

var E = require( './E.js' );
var log = require( './logging.js' ).file( '/var/log/brauerei.log' );

var repl = require( './repl.js' )( {} );

var Websocket = require( './websocket.js' ),
    websocket = false;
var Mqtt = require( './mqtt.js' ),
    mqtt = false;

var Config = require( './config.js' ),
    config = false,
	hello = false;

var Boilers = require( './boiler.js' ),
    boilers = false,
	Brewery = require( './brewery.js' ),
	brewery = false;

var State = require( './state.js' ),
    state = false;

var Ctrl = require( './ctrl.js' ),
    ctrl = false;

function initConfig( done ) {

	Config( function( err, data ) {

		if( err ) {
			log.failure( "config", err );
			return done( err );
		}

		config = data;
		hello = {
			config: {
				boilers: config.boilers,
			}
		}
		repl.addContext( { config: config, hello: hello } );

		log.startup( "config", "READY" );

		return done();

	} );
}

function initState( done ) {

	State( config.state, function( err, data ) {

		log.trace( "state", err, data );

		if( err ){
			log.failure( "state", err );
			throw new Error( err );
		}

		state = data;
		repl.addContext( { state: state } );

		setInterval( function() {

			State.save( stateSaved );

		}, 5000 );

		log.startup( "state", "READY" );

		return done();
	} );
}

function stateSaved( err ) {

	if( err ) throw new Error( err );

	log.trace( "STATE saved" );
}

function initBoilers( done ) {

	// Create something to store state in
	if( !( 'boilers' in state ) ) {
		log.trace( "create boiler state" );
		state.boilers = {};
	}

	Boilers.createAll( config.boilers, state.boilers, function( err, data ) {
	
		if( err ) {
			log.failure( "boilers" );
			return done( err );
		}

		boilers = data;

		brewery = Brewery( boilers );
		repl.addContext( { brewery: brewery } );
		repl.addContext( {
				brewery: brewery,
				boilers: boilers,
				boiler1: boilers.boiler1,
				boiler2: boilers.boiler2
		} );

		log.startup( "boilers", "READY" );

		return done();
	
	} );
}

function startWebsocket( done ) {

	Websocket( ctrl.gotWebData, hello, config.ws, function( err, data ) {

		if( err ){
			log.failure( "websockets" );
			return done( err );
		}

		websocket = data;

		ctrl.onWebMessage( websocket.send );

		log.startup( "websockets", "STARTED" );

		return done();
	} );
}

function startMqtt( done ) {

	Mqtt( ctrl.gotMqttData, config.mqtt,
			brewery.subscribe,
			function( err, data ) {

				if( err ) {
					log.failure( "mqtt", err );
					return done( err );
				}

				mqtt = data;

				ctrl.onMqttMessage( mqtt.send );

				log.startup( "mqtt", "STARTED" );

				return done();
			} );
}

function stateReady( err ) {

	if( err ) throw err;

	ctrl = Ctrl( config, hello, state, brewery );
	repl.addContext( { ctrl: ctrl } );

	async.parallel( [ startWebsocket, startMqtt ], startupDone );
}

function startupDone( err ) {

	if( err ) {
		log.failure( "Startup", err );
		throw err;
	}

	setInterval( ctrl.run, 500 );

	log.startup( "Startup", "DONE" );
}

E.cho( "Startup" );
log.startup( "Startup...", "" );

/*
// Make cleaner exit handling:
process.stdin.resume();

function gotExit( opt, err ) {

	E.rr( arguments );

	if (opt.cleanup) E.cho( 'clean' );
	if (err) E.cho( err.stack );
	if (opt.exit) process.exit();
}

//do something when app is closing
process.on( 'exit', gotExit.bind( null, { cleanup: true } ) );

//catches ctrl+c event
process.on( 'SIGINT', gotExit.bind( null, { exit: true } ) );

//catches uncaught exceptions
process.on( 'uncaughtException', gotExit.bind( null, { exit: true } ) );
*/
process.on( 'uncaughtException', function( ex ) {

	console.log( "SOME ERROR".red );
	if( ex ) console.log( ex.orange );
	console.log( util.inspect( brewery.boilers, {showHidden:false, depth: null} ) );
	if( ex && ex.stack ) console.log( ex.stack.yellow );
} );

// === Start Startup ===
async.series( [ initConfig, initState, initBoilers ], stateReady );

