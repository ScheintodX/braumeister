"use strict";

require( 'colors' );

var fs = require( 'fs' );

//This is blocking because we want logging "just to be there"

var FILE = null;
var pause = false;

var log = require( 'tracer' ).colorConsole( {
	level: 'info',
	inspectOpt: { depth: 3 },
	transport : function( data ) {
		if( pause ) return;
		if( FILE ) {
			fs.open( FILE, 'a', '0666', function( err, id ) {
				if( err ) throw new Error( err ); //fail fast!
				fs.write( id, data.output+"\n", null, 'utf8', function( err ) {
					fs.close( id );
					if( err ) throw new Error( err );
				});
			});
		} else {
			console.log( data.output );
		}
	}
} );

log.startup = function( part, state ) {
	console.log( part + ' ' + state.green );
	log.info.apply( log, arguments );
}
log.failure = function( part, err ) {
	console.log( part + ' ' + err.red );
	log.error.apply( log, arguments );
}
log.ex = function( ex ) {
	console.error.apply( console, arguments );
	log.error.apply( arguments );
	if( ex.stack ){
		console.error( ex.stack );
		log.error.apply( ex.stack );
	}
}

module.exports = log;
module.exports.file = function( file ) {
	FILE = file;
	return log;
}

module.exports.pause = function( value ) {
	pause = value;
	return log;
}
